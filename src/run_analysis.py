import typing
import tqdm

import pandas as pd

from scipy.stats import spearmanr

from common.constants import Constants


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(str(Constants.ENHANCED_DATASET))


FilterFunction: typing.TypeAlias = typing.Callable[[pd.DataFrame], pd.DataFrame]


class CorrelationData(typing.TypedDict):
    gpt_correlation: typing.List[float]
    gpt_r: typing.List[float]
    bert_correlation: typing.List[float]
    bert_r: typing.List[float]
    llama_correlation: typing.List[float]
    llama_r: typing.List[float]
    participant: typing.List[str]
    column: typing.List[str]


class Correlation(typing.NamedTuple):
    correlation: float
    r: float


class Correlations(typing.NamedTuple):
    gpt_correlation: Correlation
    bert_correlation: Correlation
    llama_correlation: Correlation


def run_correlation(
    df: pd.DataFrame,
    target_column: typing.Literal[
        "RECORDING_SESSION_LABEL",
        "Participant_ID",
        "Word_Unique_ID",
        "Text_ID",
        "Word_Number",
        "Sentence_Number",
        "Word_In_Sentence_Number",
        "Word",
        "Word_Cleaned",
        "Word_Length",
        "Total_Response_Count",
        "Unique_Count",
        "OrthographicMatch",
        "OrthoMatchModel",
        "IsModalResponse",
        "ModalResponse",
        "ModalResponseCount",
        "Certainty",
        "POS_CLAWS",
        "Word_Content_Or_Function",
        "Word_POS",
        "POSMatch",
        "POSMatchModel",
        "InflectionMatch",
        "InflectionMatchModel",
        "LSA_Context_Score",
        "LSA_Response_Match_Score",
        "IA_ID",
        "IA_LABEL",
        "TRIAL_INDEX",
        "IA_LEFT",
        "IA_RIGHT",
        "IA_TOP",
        "IA_BOTTOM",
        "IA_AREA",
        "IA_FIRST_FIXATION_DURATION",
        "IA_FIRST_FIXATION_INDEX",
        "IA_FIRST_FIXATION_VISITED_IA_COUNT",
        "IA_FIRST_FIXATION_X",
        "IA_FIRST_FIXATION_Y",
        "IA_FIRST_FIX_PROGRESSIVE",
        "IA_FIRST_FIXATION_RUN_INDEX",
        "IA_FIRST_FIXATION_TIME",
        "IA_FIRST_RUN_DWELL_TIME",
        "IA_FIRST_RUN_FIXATION_COUNT",
        "IA_FIRST_RUN_END_TIME",
        "IA_FIRST_RUN_FIXATION_.",
        "IA_FIRST_RUN_START_TIME",
        "IA_DWELL_TIME",
        "IA_FIXATION_COUNT",
        "IA_RUN_COUNT",
        "IA_SKIP",
        "IA_REGRESSION_IN",
        "IA_REGRESSION_IN_COUNT",
        "IA_REGRESSION_OUT",
        "IA_REGRESSION_OUT_COUNT",
        "IA_REGRESSION_OUT_FULL",
        "IA_REGRESSION_OUT_FULL_COUNT",
        "IA_REGRESSION_PATH_DURATION",
        "IA_FIRST_SACCADE_AMPLITUDE",
        "IA_FIRST_SACCADE_ANGLE",
        "IA_FIRST_SACCADE_END_TIME",
        "IA_FIRST_SACCADE_START_TIME",
    ],
    filter_function: typing.Optional[FilterFunction] = None,
) -> Correlations:
    df_filtered = df.dropna(
        subset=[
            target_column,
            Constants.BERT_SURPRISAL_COLUMN,
            Constants.GPT_SURPRISAL_COLUMN,
        ]
    )
    if filter_function is not None:
        df_filtered = df_filtered[filter_function(df_filtered)]

    gpt_correlation = Correlation(
        *spearmanr(
            df_filtered[Constants.GPT_SURPRISAL_COLUMN], df_filtered[target_column]
        )
    )
    bert_correlation = Correlation(
        *spearmanr(
            df_filtered[Constants.BERT_SURPRISAL_COLUMN], df_filtered[target_column]
        )
    )
    llama_correlation = Correlation(
        *spearmanr(
            df_filtered[Constants.LLAMA_SURPRISAL_COLUMN], df_filtered[target_column]
        )
    )
    return Correlations(
        gpt_correlation=gpt_correlation,
        bert_correlation=bert_correlation,
        llama_correlation=llama_correlation,
    )


def add_record(
    data: CorrelationData,
    result: Correlations,
    column: str,
    participant_id: str = "ALL",
):
    data["bert_correlation"].append(result.bert_correlation.correlation)
    data["bert_r"].append(result.bert_correlation.r)

    data["gpt_correlation"].append(result.gpt_correlation.correlation)
    data["gpt_r"].append(result.gpt_correlation.r)

    data["llama_correlation"].append(result.llama_correlation.correlation)
    data["llama_r"].append(result.llama_correlation.r)

    data["column"].append(column)
    data["participant"].append(participant_id)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in [
        "IA_FIRST_FIXATION_DURATION",
        "IA_FIRST_FIXATION_INDEX",
        "IA_FIRST_FIXATION_VISITED_IA_COUNT",
        "IA_FIRST_FIXATION_TIME",
        "IA_FIRST_RUN_DWELL_TIME",
        "IA_FIRST_RUN_FIXATION_COUNT",
        "IA_FIRST_RUN_END_TIME",
        "IA_FIRST_RUN_FIXATION_.",
        "IA_DWELL_TIME",
        "IA_FIXATION_COUNT",
        "IA_RUN_COUNT",
        "IA_SKIP",
        "IA_REGRESSION_IN",
        "IA_REGRESSION_IN_COUNT",
        "IA_REGRESSION_OUT",
        "IA_REGRESSION_OUT_COUNT",
        "IA_REGRESSION_OUT_FULL",
        "IA_REGRESSION_OUT_FULL_COUNT",
        "IA_REGRESSION_PATH_DURATION",
        "IA_FIRST_SACCADE_AMPLITUDE",
        "IA_FIRST_SACCADE_ANGLE",
    ]:
        modified = df.groupby("Participant_ID")[column].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        df[f"{column}_Z"] = modified

    return df


def main():
    df = load_dataset()
    df = normalize_columns(df)
    results: CorrelationData = {
        "bert_correlation": [],
        "bert_r": [],
        "gpt_correlation": [],
        "gpt_r": [],
        "llama_correlation": [],
        "llama_r": [],
        "column": [],
        "participant": [],
    }
    for column in tqdm.tqdm(list(filter(lambda x: x.startswith("IA"), df.columns))):
        add_record(results, run_correlation(df, column), column)

    pd.DataFrame(results).to_csv("all_participant_correlations.csv")

    print("Starting per participant correlations")
    for pid, group in tqdm.tqdm(df.groupby("Participant_ID")):
        for column in filter(lambda x: x.startswith("IA"), df.columns):
            add_record(results, run_correlation(group, column), column, pid)

    pd.DataFrame(results).to_csv("participant_level_correlations.csv")


if __name__ == "__main__":
    main()
