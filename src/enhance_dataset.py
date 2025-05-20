import pathlib
import tqdm

import pandas as pd


from bert_surprisal import bert_surprisal
from gpt_surprisal import gpt_surprisal
from llama_surprisal import llama_surprisal

DATASET_DIRECTORY = pathlib.Path(__file__).parent.parent / "datasets"


def read_dataset() -> pd.DataFrame:
    return pd.read_csv(str(DATASET_DIRECTORY / "provo.csv"))


def write_dataset(df: pd.DataFrame):
    return df.to_csv(str(DATASET_DIRECTORY / "provo.enhanced.csv"))


def enhance(df: pd.DataFrame) -> pd.DataFrame:
    enhancements = {
        "Word_Unique_ID": [],
        "gpt_surprisal": [],
        "bert_surprisal": [],
        "llama_surprisal": [],
    }
    for _, group in tqdm.tqdm(
        (
            df.drop_duplicates(subset=["Word_Unique_ID"])
            .sort_values(["Text_ID", "Sentence_Number", "Word_In_Sentence_Number"])
            .groupby(["Text_ID", "Sentence_Number"])
        )
    ):
        sentence = group["Word_Cleaned"].tolist()
        enhancements["gpt_surprisal"].extend(gpt_surprisal(sentence))
        enhancements["bert_surprisal"].extend(bert_surprisal(sentence))
        enhancements["llama_surprisal"].extend(llama_surprisal(sentence))
        enhancements["Word_Unique_ID"].extend(group["Word_Unique_ID"].tolist())

    print("Done! now merging")
    return df.merge(pd.DataFrame(enhancements), how="left", on="Word_Unique_ID")


def clean_df(df: pd.DataFrame):
    cleaned = df.dropna(subset=["Word_Unique_ID"])
    missing = cleaned["Sentence_Number"].isna()
    cleaned.loc[missing, "Sentence_Number"] = 1
    cleaned.loc[missing, "Word_In_Sentence_Number"] = cleaned.loc[
        missing, "Word_Number"
    ]
    return cleaned


def main():
    df = read_dataset()
    df = clean_df(df)
    enhanced = enhance(df)
    write_dataset(enhanced)


if __name__ == "__main__":
    main()
