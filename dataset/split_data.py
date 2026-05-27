import pandas as pd
from sklearn.model_selection import train_test_split

input_csv = "/home/trandat/Documents/gipformer/dataset/data.gipformer.csv"

df = pd.read_csv(input_csv)

# giữ lại các dòng KHÔNG chứa số
mask = ~df["text"].astype(str).str.contains(r"\d", regex=True)
df = df[mask].copy()

# train = 70%
# temp = 30%
train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    shuffle=True,
)

# temp 30% -> test 20%, dev 10%
# tức là:
# test = 2/3 của temp
# dev = 1/3 của temp
# test_df, dev_df = train_test_split(
#     temp_df,
#     test_size=1/3,
#     random_state=42,
#     shuffle=True,
# )

# save
# breakpoint()
train_df.to_csv(input_csv.replace(".csv",".train.csv"), index=False)
test_df.to_csv(input_csv.replace(".csv",".test.csv"), index=False)
# dev_df.to_csv("dev.csv", index=False)

print("train:", len(train_df))
print("test:", len(test_df))
# print("dev:", len(dev_df))