import argparse
import inspect
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from lhotse import (
    CutSet,
    Fbank,
    FbankConfig,
    load_manifest_lazy,
)
from lhotse.dataset import (
    CutConcatenate,
    DynamicBucketingSampler,
    K2SpeechRecognitionDataset,
    SimpleCutSampler,
    SpecAugment,
    PrecomputedFeatures,
)
from lhotse.dataset.input_strategies import OnTheFlyFeatures
from lhotse.utils import fix_random_seed
from lhotse.cut import Cut

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class CustomAsrDataModule:

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):

        group = parser.add_argument_group(
            title="Custom ASR data options",
        )

        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/manifests"),
        )

        group.add_argument(
            "--max-duration",
            type=float,
            default=40.0,
        )

        group.add_argument(
            "--bucketing-sampler",
            type=str2bool,
            default=True,
        )

        group.add_argument(
            "--num-buckets",
            type=int,
            default=30,
        )

        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
        )

        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
        )

        group.add_argument(
            "--gap",
            type=float,
            default=1.0,
        )

        group.add_argument(
            "--on-the-fly-feats",
            type=str2bool,
            default=False,
        )

        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
        )

        group.add_argument(
            "--drop-last",
            type=str2bool,
            default=True,
        )

        group.add_argument(
            "--return-cuts",
            type=str2bool,
            default=True,
        )

        group.add_argument(
            "--num-workers",
            type=int,
            default=2,
        )

        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=True,
        )

        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
        )

        group.add_argument(
            "--input-strategy",
            type=str,
            default="PrecomputedFeatures",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:

        transforms = []

        if self.args.concatenate_cuts:

            logging.info(
                f"Using cut concatenation "
                f"(duration_factor={self.args.duration_factor}, "
                f"gap={self.args.gap})"
            )

            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor,
                    gap=self.args.gap,
                )
            ]

        input_transforms = []

        if self.args.enable_spec_aug:

            logging.info("Enable SpecAugment")

            num_frame_masks = 10

            num_frame_masks_parameter = inspect.signature(
                SpecAugment.__init__
            ).parameters["num_frame_masks"]

            if num_frame_masks_parameter.default == 1:
                num_frame_masks = 2

            input_transforms.append(
                SpecAugment(
                    time_warp_factor=self.args.spec_aug_time_warp_factor,
                    num_frame_masks=num_frame_masks,
                    features_mask_size=27,
                    num_feature_masks=2,
                    frames_mask_size=100,
                )
            )

        if self.args.on_the_fly_feats:

            train = K2SpeechRecognitionDataset(
                cut_transforms=transforms,

                input_strategy=OnTheFlyFeatures(
                    Fbank(FbankConfig(num_mel_bins=80))
                ),

                input_transforms=input_transforms,
                return_cuts=self.args.return_cuts,
            )

        else:

            train = K2SpeechRecognitionDataset(
                input_strategy=eval(self.args.input_strategy)(),
                cut_transforms=transforms,
                input_transforms=input_transforms,
                return_cuts=self.args.return_cuts,
            )

        if self.args.bucketing_sampler:

            train_sampler = DynamicBucketingSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                buffer_size=self.args.num_buckets * 5000,
                drop_last=self.args.drop_last,
            )

        else:

            train_sampler = SimpleCutSampler(
                cuts_train,
                max_duration=self.args.max_duration,
                shuffle=self.args.shuffle,
            )

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(
                sampler_state_dict
            )

        seed = torch.randint(0,100000,(),).item()

        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:

        transforms = []

        if self.args.concatenate_cuts:
            transforms = [
                CutConcatenate(
                    duration_factor=self.args.duration_factor,
                    gap=self.args.gap,
                )
            ]

        if self.args.on_the_fly_feats:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=OnTheFlyFeatures(
                    Fbank(FbankConfig(num_mel_bins=80))
                ),
                return_cuts=self.args.return_cuts,
            )
        else:
            validate = K2SpeechRecognitionDataset(
                cut_transforms=transforms,
                input_strategy=eval(self.args.input_strategy)(),
                return_cuts=self.args.return_cuts,
            )

        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
        )

        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
        )

        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:

        test = K2SpeechRecognitionDataset(
            input_strategy=(
                OnTheFlyFeatures(
                    Fbank(
                        FbankConfig(num_mel_bins=80)
                    )
                )
                if self.args.on_the_fly_feats
                else eval(self.args.input_strategy)()
            ),

            return_cuts=self.args.return_cuts,
        )

        sampler = DynamicBucketingSampler(
            cuts,
            max_duration=self.args.max_duration,
            shuffle=False,
        )

        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )

        return test_dl

    @lru_cache()
    def load_manifest(self, manifest_filename: str) -> CutSet:
        manifest_path = self.args.manifest_dir / manifest_filename
        logging.info(f"Loading manifest: {manifest_path}")
        return load_manifest_lazy(manifest_path)

    @lru_cache()
    def train_cuts(self) -> CutSet:
        return self.load_manifest("train.jsonl.gz")

    @lru_cache()
    def valid_cuts(self) -> CutSet:
        return self.load_manifest("test.jsonl.gz")

    @lru_cache()
    def test_cuts(self) -> CutSet:
        return self.load_manifest("test.jsonl.gz")
    
def remove_short_and_long_utt(sp ,c: Cut):
    # Keep only utterances with duration between 1 second and 20 seconds
    #
    # Caution: There is a reason to select 20.0 here. Please see
    # ../local/display_manifest_statistics.py
    #
    # You should use ../local/display_manifest_statistics.py to get
    # an utterance duration distribution for your dataset to select
    # the threshold
    if c.duration < 1.0 or c.duration > 30.0:
        # logging.warning(
        #     f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
        # )
        return False

    # In pruned RNN-T, we require that T >= S
    # where T is the number of feature frames after subsampling
    # and S is the number of tokens in the utterance

    # In ./zipformer.py, the conv module uses the following expression
    # for subsampling
    T = ((c.num_frames - 7) // 2 + 1) // 2
    tokens = sp.encode(c.supervisions[0].text, out_type=str)

    # breakpoint()

    if T < len(tokens):
        logging.warning(
            f"Exclude cut with ID {c.id} from training. "
            f"Number of frames (before subsampling): {c.num_frames}. "
            f"Number of frames (after subsampling): {T}. "
            f"Text: {c.supervisions[0].text}. "
            f"Tokens: {tokens}. "
            f"Number of tokens: {len(tokens)}"
        )
        return False

    return True


