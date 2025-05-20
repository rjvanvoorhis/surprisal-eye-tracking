import pathlib
import typing
import tqdm
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

from scipy.stats import spearmanr, pearsonr


from common.constants import Constants


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(str(Constants.ENHANCED_DATASET))


FilterFunction: typing.TypeAlias = typing.Callable[[pd.DataFrame], pd.DataFrame]


PLOTS_FOLDER = pathlib.Path(__file__).parent.parent / "plots"
PLOTS_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER = pathlib.Path(__file__).parent.parent / "results"
RESULTS_FOLDER.mkdir(exist_ok=True)


def prune_data(data: pd.DataFrame, columns: typing.List[str]) -> pd.DataFrame:
    return data[columns].dropna().reset_index(drop=True)


def add_z_column(df: pd.DataFrame, column: str, fillna: bool = True) -> pd.DataFrame:
    print(f"Before {df.shape}")
    if fillna:
        df[column] = df[column].fillna(value=0.0)
    else:
        df = df.dropna(subset=[column]).reset_index(drop=True)
    df[f"{column}_Z"] = df.groupby("Participant_ID")[column].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    print(f"After: {df.shape}")
    return df


def enhance_data(df: pd.DataFrame) -> pd.DataFrame:
    for column in [
        "IA_DWELL_TIME",
        "IA_REGRESSION_PATH_DURATION",
        "IA_FIRST_FIXATION_DURATION",
    ]:
        df = add_z_column(df, column)
    return df


def plot_multi_participant(df: pd.DataFrame, by: str, against: str, size: int = 6):
    np.random.seed(42)
    selected = np.random.choice(
        df["Participant_ID"].dropna().unique(), size=size, replace=False
    )
    df = prune_data(df, ["Participant_ID", by, against])
    sample_df = df[df["Participant_ID"].isin(selected)]
    plt.figure(figsize=(10, 6))
    sns.lmplot(
        data=sample_df,
        x=against,
        y=by,
        hue="Participant_ID",
        col="Participant_ID",
        col_wrap=3,
        height=4,
        scatter_kws={"alpha": 0.2},
        line_kws={"linewidth": 2},
        facet_kws={"sharex": True, "sharey": True},
        ci=95,
    )
    plt.subplots_adjust(top=0.9)
    plt.tight_layout()
    filename = str(PLOTS_FOLDER / f"{by}-vs-{against}-by-participant.png")
    plt.savefig(filename)


def plot_random_effects_slopes(result, filename):
    re_df = pd.DataFrame(result.random_effects).T.reset_index()
    re_df.columns = ["Participant_ID", "Intercept", "Slope"]

    re_df = re_df.sort_values("Slope")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="Intercept", y="Slope", data=re_df)
    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.title("Random Intercepts and Slopes by Participant")
    plt.xlabel("Random Intercept")
    plt.ylabel("Random Slope for Predictor")
    plt.tight_layout()

    out_path: pathlib.Path = PLOTS_FOLDER / filename
    plt.savefig(str(out_path.absolute()))
    plt.close()


def plot_predicted_vs_observed(
    df: pd.DataFrame, by: str, filename: str, size: int = 5000
):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=df.sample(min(size, len(df))), x="Predicted", y=by, alpha=0.3)
    plt.plot(
        [df["Predicted"].min(), df["Predicted"].max()],
        [df["Predicted"].min(), df["Predicted"].max()],
        linestyle="--",
        color="red",
    )
    plt.title("Predicted vs Observed")
    plt.tight_layout()
    plt.savefig(str(PLOTS_FOLDER / filename))
    plt.close()


def plot_mixed(df: pd.DataFrame, by: str, against: str, sample_size: int = 5000):
    df = prune_data(df, [by, against, "Participant_ID"])

    model = smf.mixedlm(
        f"{by} ~ {against}",
        data=df,
        groups=df["Participant_ID"],
        re_formula=f"~{against}",
    )
    result = model.fit()

    df["Predicted"] = result.fittedvalues
    df["Residuals"] = result.resid

    suffix = f"{by}-vs-{against}.png"
    plot_random_effects_slopes(result, f"slopes-{suffix}")
    plot_predicted_vs_observed(df, by, f"predicted-{suffix}", sample_size)
    plot_multi_participant(df, by, against, 6)
    return extract_model_info(result, by, against, transform=None, model_type="linear")


def analyze_binary_predictor(df: pd.DataFrame, by: str, against: str):
    df = prune_data(df, [by, against])
    df[by] = df[by].astype(int)

    model = smf.glm(f"{by} ~ {against}", data=df, family=sm.families.Binomial())

    try:
        result = model.fit()
        if np.isinf(result.params[against]) or np.isnan(result.bse[against]):
            raise ValueError("Infinite or invalid coefficient, using regularized fit.")
    except Exception:
        result = model.fit_regularized(alpha=1.0, L1_wt=0.0)
        # Provide a dummy value for SE, z, p to indicate regularized
        return {
            "variable": by,
            "model_type": "logistic",
            "predictor": against,
            "coef": result.params[against],
            "se": None,
            "z": None,
            "p": None,
            "exp_coef": np.exp(result.params[against]),
        }

    return extract_model_info(
        result, by, against, transform=np.exp, model_type="logistic"
    )


def analyze_count_predictor(df: pd.DataFrame, by: str, against: str):
    df = df[[by, against]].dropna()
    model = smf.glm(f"{by} ~ {against}", data=df, family=sm.families.Poisson())
    result = model.fit()
    info = extract_model_info(
        result, by, against, transform=np.exp, model_type="poisson"
    )
    return info


def plot_logistic(df: pd.DataFrame, by: str, against: str):
    df = prune_data(df, [by, against])
    plt.figure(figsize=(6, 4))
    sns.regplot(
        x=against, y=by, data=df, logistic=True, ci=95, scatter_kws={"alpha": 0.2}
    )
    plt.title(f"Logistic Regression: {by} vs {against}")
    plt.tight_layout()
    filename = f"logistic-{by}-vs-{against}.png"
    plt.savefig(str(PLOTS_FOLDER / filename))
    plt.close()


def plot_poisson(df: pd.DataFrame, by: str, against: str):
    df = prune_data(df, [by, against])
    model = smf.glm(f"{by} ~ {against}", data=df, family=sm.families.Poisson())
    result = model.fit()
    df["Predicted"] = result.fittedvalues
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=against, y=by, data=df, alpha=0.3)
    sns.lineplot(x=against, y="Predicted", data=df.sort_values(against), color="red")
    plt.title(f"Poisson Regression: {by} vs {against}")
    plt.tight_layout()
    filename = f"poisson-{by}-vs-{against}.png"
    plt.savefig(str(PLOTS_FOLDER / filename))
    plt.close()


def extract_model_info(
    result, variable, predictor, transform=np.exp, model_type="linear"
):
    coef = result.params[predictor]
    se = result.bse[predictor]
    z = coef / se
    p = result.pvalues[predictor]
    return {
        "model_type": model_type,
        "predictor": predictor,
        "coef": coef,
        "se": se,
        "z": z,
        "p": p,
        "variable": variable,
        "exp_coef": transform(coef) if transform else None,
    }


def main():
    df = load_dataset()
    df = enhance_data(df)
    results = []
    results_file = RESULTS_FOLDER / "regression_data.csv"
    for model in ["gpt_surprisal", "bert_surprisal"]:
        for column in [
            "IA_DWELL_TIME",
            "IA_DWELL_TIME_Z",
            "IA_FIRST_FIXATION_DURATION",
            "IA_FIRST_FIXATION_DURATION_Z",
            "IA_REGRESSION_PATH_DURATION",
            "IA_REGRESSION_PATH_DURATION_Z",
        ]:
            results.append(plot_mixed(df, column, model))
            pd.DataFrame.from_records(results).to_csv(str(results_file))

        for binary_column in ["IA_SKIP", "IA_REGRESSION_IN", "IA_REGRESSION_OUT"]:
            results.append(analyze_binary_predictor(df, binary_column, model))
            pd.DataFrame.from_records(results).to_csv(str(results_file))

        for count_column in ["IA_REGRESSION_IN_COUNT", "IA_REGRESSION_OUT_COUNT"]:
            results.append(analyze_count_predictor(df, count_column, model))
            pd.DataFrame.from_records(results).to_csv(str(results_file))
        #     plot_poisson(df, count_col, model)


if __name__ == "__main__":
    main()
