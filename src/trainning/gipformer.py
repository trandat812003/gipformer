'''
export PYTHONPATH=/home/trandat/Documents/gipformer/icefall:/home/trandat/Documents/gipformer/egs:$PYTHONPATH

# Fine-tune without mux (i.e not mixing with original training data):
python -m src.trainning.gipformer \
  --world-size 1 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --use-ctc 0 \
  --do-finetune 1 \
  --finetune-ckpt /media/trandat/Data/model/gipformer/epoch-35-avg-6.pt \
  --bpe-model /media/trandat/Data/model/gipformer/bpe.model \
  --manifest-dir /home/trandat/Documents/gipformer/data/manifests \
  --base-lr 0.0045 \
  --use-mux 0 \
  --exp-dir zipformer/exp_finetune \
  --max-duration 40
'''

import logging
import copy
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import sentencepiece as spm
from lhotse.utils import fix_random_seed

from icefall import diagnostics
from icefall.dist import cleanup_dist, setup_dist
from icefall.hooks import register_inf_check_hooks
from icefall.utils import (
    create_grad_scaler,
    get_parameter_groups_with_lrs,
    setup_logger,
)
from egs.librispeech.ASR.zipformer.optim import Eden, ScaledAdam
from egs.librispeech.ASR.zipformer.asr_datamodule import LibriSpeechAsrDataModule
from egs.librispeech.ASR.zipformer.finetune import (
    get_parser,
    get_params, 
    load_model_params, 
    load_checkpoint_if_available, 
    scan_pessimistic_batches_for_oom, 
    train_one_epoch, 
    save_checkpoint,
)

from src.models.gipformer import load_model
from src.trainning.data_module import CustomAsrDataModule, remove_short_and_long_utt


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_params()
    params.update(vars(args))

    # breakpoint()

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    model, sp, fbank, device = load_model()

    # <blk> is defined in local/train_bpe_model.py
    params.blank_id = sp.piece_to_id("<blk>")
    params.vocab_size = sp.get_piece_size()

    if not params.use_transducer:
        params.ctc_loss_scale = 1.0

    logging.info(params)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of model parameters: {num_param}")

    assert params.save_every_n >= params.average_period
    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    # load model parameters for model fine-tuning
    if params.do_finetune:
        assert params.start_epoch == 1, "Fine-tune must start from epoch 1"
        modules = params.init_modules.split(",") if params.init_modules else None
        checkpoints = load_model_params(
            ckpt=params.finetune_ckpt, model=model, init_modules=modules, strict=False
        )
        # Need to update the model_avg if use initialisation
        if rank == 0:
            # model_avg is only used with rank 0
            model_avg = copy.deepcopy(model).to(torch.float64)
    else:
        # resuming training
        assert params.start_epoch > 1, params.start_epoch
        checkpoints = load_checkpoint_if_available(
            params=params, model=model, model_avg=model_avg
        )

    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(model, lr=params.base_lr, include_names=True),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )

    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)

    if checkpoints and "optimizer" in checkpoints:
        logging.info("Loading optimizer state dict")
        optimizer.load_state_dict(checkpoints["optimizer"], strict=False)

    if (
        checkpoints
        and "scheduler" in checkpoints
        and checkpoints["scheduler"] is not None
    ):
        logging.info("Loading scheduler state dict")
        scheduler.load_state_dict(checkpoints["scheduler"], strict=False)

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    datamodule = CustomAsrDataModule(args)

    train_cuts = datamodule.train_cuts()
    logging.info(train_cuts)

    train_cuts = train_cuts.filter(lambda c: remove_short_and_long_utt(sp, c))

    if params.start_batch > 0 and checkpoints and "sampler" in checkpoints:
        # We only load the sampler's state dict when it loads a checkpoint
        # saved in the middle of an epoch
        sampler_state_dict = checkpoints["sampler"]
    else:
        sampler_state_dict = None

    train_dl = datamodule.train_dataloaders(
        train_cuts,
        sampler_state_dict=sampler_state_dict,
    )

    valid_cuts = datamodule.valid_cuts()
    valid_sets = ["valid"]
    valid_dls = [datamodule.valid_dataloaders(valid_cuts)]

    if not params.print_diagnostics:
        scan_pessimistic_batches_for_oom(
            model=model,
            train_dl=train_dl,
            optimizer=optimizer,
            sp=sp,
            params=params,
        )

    scaler = create_grad_scaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"], strict=False)

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        scheduler.step_epoch(epoch - 1)
        fix_random_seed(params.seed + epoch - 1)
        train_dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sp=sp,
            train_dl=train_dl,
            valid_dls=valid_dls,
            valid_sets=valid_sets,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        save_checkpoint(
            params=params,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=train_dl.sampler,
            scaler=scaler,
            rank=rank,
        )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    CustomAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)

if __name__ == "__main__":
    main()