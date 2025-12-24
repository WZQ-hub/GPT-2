import urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "../data/sms_spam_collection.zip"
extracted_path = Path("../data/sms_spam_collection")
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download "
              "and extraction."
              )
        return
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

# download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
df = pd.read_csv(
data_file_path, sep="\t", header=None, names=["Label", "Text"]
)

def create_balanced_dataset(df):
    # 1. 计算数据集中 "spam"（垃圾邮件）的总行数
    # df["Label"] == "spam" 返回布尔序列，.shape[0] 获取行数
    num_spam = df[df["Label"] == "spam"].shape[0]

    # 2. 从 "ham"（正常邮件）中随机抽取与 spam 数量相等的样本
    # sample 函数用于随机抽样
    # n=num_spam 指定抽取的数量
    # random_state=123 是随机种子，确保每次运行代码抽到的数据是一样的（可复现性）
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )

    # 3. 使用 concat 将抽取的 ham 子集和所有的 spam 拼接在一起
    # pd.concat 默认按行拼接（axis=0）
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ])

    return balanced_df

balanced_df = create_balanced_dataset(df)

balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
# print(balanced_df["Label"].value_counts())

def random_split(df, train_frac, val_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    val_end = int(len(df) * val_frac) + train_end
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    return train_df, val_df, test_df
train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

train_path = extracted_path / "train.csv"
validation_path = extracted_path / "validation.csv"
test_path = extracted_path / "test.csv"

train_df.to_csv(train_path, index=False)
validation_df.to_csv(validation_path, index=False)
test_df.to_csv(test_path, index=False)

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
print(f"训练集行数: {len(train_df)}")
print(f"验证集行数: {len(validation_df)}")
print(f"测试集行数: {len(test_df)}")