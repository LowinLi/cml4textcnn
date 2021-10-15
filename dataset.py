from torchtext.data import TabularDataset, Field
import jieba_fast
import json
import pandas as pd


def get_iflytek_dataset(max_length):
    sentence = Field(
        include_lengths=True,
        fix_length=max_length,
        tokenize=jieba_fast.lcut,
    )
    label = Field(sequential=False, unk_token=None)
    for file in ["train", "dev"]:
        records = []
        with open(f"dataset/{file}.json", "r") as f:
            for line in f.readlines():
                records.append(json.loads(line))
        df = pd.DataFrame(records)
        df = df[["sentence", "label"]]
        df.to_csv(f"dataset/{file}.tsv", sep="\t", index=False)
    train = TabularDataset(
        path="dataset/train.tsv",
        format="tsv",
        skip_header=True,
        fields=[("sentence", sentence), ("label", label)],
    )
    dev = TabularDataset(
        path="dataset/dev.tsv",
        format="tsv",
        skip_header=True,
        fields=[("sentence", sentence), ("label", label)],
    )
    sentence.build_vocab(train)
    label.build_vocab(train)

    return sentence, label, train, dev


if __name__ == "__main__":
    sentence, label, train, dev = get_iflytek_dataset()
