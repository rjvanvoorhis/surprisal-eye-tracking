import pathlib
import typing

DATASET_COLUMN: typing.TypeAlias = typing.Literal[
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
]


class Constants:
    RAW_DATASET: pathlib.Path = (
        pathlib.Path(__file__).parent.parent.parent / "datasets" / "provo.csv"
    )
    ENHANCED_DATASET = RAW_DATASET.parent / "provo.enhanced.csv"
    GPT_MODEL_NAME = "gpt2"
    BERT_MODEL_NAME = "bert-base-uncased"
    LLAMA_MODEL_NAME = "baffo32/decapoda-research-llama-7B-hf"
    GPT_SURPRISAL_COLUMN = "gpt_surprisal"
    BERT_SURPRISAL_COLUMN = "bert_surprisal"
    LLAMA_SURPRISAL_COLUMN = "llama_surprisal"
    DATASET_COLUMN = DATASET_COLUMN
