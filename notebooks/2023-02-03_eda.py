# # EDA for landing distance modeling competition

import math
import random
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdpipe as pdp
from lazypredict.Supervised import LazyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

desired_width = 320
pd.set_option("display.width", desired_width)
pd.set_option("display.max_columns", 16)
np.set_printoptions(linewidth=desired_width)

data_path = Path(r"C:\Nayef\icarus\data")
dst_path = Path(r"C:\Nayef\icarus\dst")

random.seed(2023)

# ## Read in data

df_adsb = pd.read_csv(data_path.joinpath("adsb_train.csv"))
keys = df_adsb["key"].unique().tolist()
train_proportion = 0.8
train_keys = random.sample(keys, math.ceil(len(keys) * train_proportion))
test_keys = list(set(keys) - set(train_keys))

# df for training and CV-based training:
df_adsb_train = df_adsb.query("key in @train_keys")
df_adsb_validation = df_adsb.query("key in @test_keys")  # for model selection
df_adsb_test = pd.read_csv(data_path.joinpath("adsb_test.csv"))  # final model only

# todo: use altitude in training and final model inference
df_altitude = pd.read_csv(data_path.joinpath("train_altitude_at_landing.csv"))

# df for training and CV-based training:
df_altitude_train = df_altitude.query("key in @train_keys")
df_altitude_validation = df_altitude.query("key in @test_keys")  # for model selection
# final model only:
df_altitude_test = pd.read_csv(data_path.joinpath("train_altitude_at_landing.csv"))

# labels datasets:
df_qar = pd.read_csv(data_path.joinpath("qar_train.csv"))

# df for training and CV-based training:
df_qar_train = df_qar.query("key in @train_keys")
df_qar_validation = df_qar.query("key in @test_keys")  # for model selection
df_qar_test = "haha, that would be too easy"  # doesn't exist


# ## Pipelines

pipe_adsb = pdp.PdPipeline(
    [
        pdp.ColumnDtypeEnforcer(
            {"key": pd.StringDtype(), "last_update": pd.StringDtype()}
        ),
        pdp.ApplyByCols("last_update", pd.to_datetime),
    ]
)
df_adsb_train = pipe_adsb(df_adsb_train)
df_adsb_train.info()
df_adsb_train.describe(include="all").T
df_adsb_validation = pipe_adsb(df_adsb_validation)
df_adsb_test = pipe_adsb(df_adsb_test)

pipe_altitude = pdp.PdPipeline([pdp.ColumnDtypeEnforcer({"key": pd.StringDtype()})])
df_altitude_train = pipe_altitude(df_altitude_train)
df_altitude_train.info()
df_altitude_train.describe(include="all").T
df_altitude_validation = pipe_altitude(df_altitude_validation)
df_altitude_test = pipe_altitude(df_altitude_test)

pipe_qar = pdp.PdPipeline([pdp.ColumnDtypeEnforcer({"key": pd.StringDtype()})])
df_qar_train = pipe_qar(df_qar_train)
df_qar_train.info()
df_qar_train.describe(include="all").T
df_qar_validation = pipe_qar(df_qar_validation)


# ## Join train datasets

df_join = df_adsb_train.merge(df_qar_train, on="key").merge(df_altitude_train, on="key")
df_join.info()

rerun_eda = False
if rerun_eda:
    x = df_join.groupby("key").size()
    assert min(x) == 15
    txt = "Distribution of row count per key"
    x.sort_values().plot.hist(title="txt")
    plt.show()

    x1 = df_join.query("on_ground == 1")
    err_txt = "There is a flight that was on-ground before reaching runway"
    assert min(x1["distance_from_threshold"]) < 0, err_txt

    x1 = x1.groupby("key").size()
    assert min(x1) == 3
    txt = "Distribution of row count per key \nFilter: on_ground=1"
    x1.sort_values().plot.hist(title=txt)
    plt.show()

    x2 = df_join.query("on_ground == 0")
    x2 = x2.groupby("key").size()
    assert min(x2) == 5
    txt = "Distribution of row count per key \nFilter: on_ground=0"
    x2.sort_values().plot.hist(title=txt)
    plt.show()


# ## Set up X df

cols_X = [
    "key",
    "distance_from_threshold",
    "track",
    "geometric_vertical_rate",
    "geometric_height",
]


def pull_values_by_index_from_on_ground(
    df: pd.DataFrame, cols: List = None
) -> pd.DataFrame:
    """
    This function takes `df` and converts values in `cols` from long to wide. It picks
    four sequential values in each col, and converts to four cols. In each case, the
    values are picked relative to the row where col `on_ground` first takes value 1.
    """

    result = {}
    for key in df["key"].unique():
        tmp = df.query("key == @key").sort_values("last_update")
        touchdown_idx = (tmp["on_ground"].values == 1).argmax()
        idx = [touchdown_idx - 2, touchdown_idx - 1, touchdown_idx, touchdown_idx + 1]

        vals = []
        for col in cols:
            if col == "key":
                continue
            vals_tmp = tmp[col].iloc[idx].reset_index(drop=True)
            vals.append(vals_tmp.values.tolist())

        vals = list(chain(*vals))
        result[key] = vals

    cols_suffixed = add_col_suffixes(cols)

    df = (
        pd.DataFrame(result.values(), index=result.keys(), columns=cols_suffixed)
        .reset_index()
        .rename(columns={"index": "key"})
    )

    return df


def add_col_suffixes(cols: List = None) -> List:
    suffixes = ["_minus2", "_minus1", "_0", "_plus1"]
    cols_suffixed = []
    for col in cols:
        if col == "key":
            continue
        tmp = [f"{col}{suff}" for suff in suffixes]
        cols_suffixed.append(tmp)
    cols_suffixed = list(chain(*cols_suffixed))
    return cols_suffixed


def drop_cols_for_training(df_train: pd.DataFrame) -> pd.DataFrame:
    # todo: retain `key` as the index
    df = df_train.drop(
        columns=[
            "track_0",
            "track_plus1",
            "geometric_vertical_rate_0",
            "geometric_vertical_rate_plus1",
            "geometric_height_0",
            "geometric_height_plus1",
        ],
    ).set_index("key")
    return df


# train data
df_X_train = pull_values_by_index_from_on_ground(df_adsb_train, cols_X)
df_train = df_X_train.merge(df_qar_train, on="key")
df_X_train = drop_cols_for_training(df_X_train)
df_y_train = df_train["touchdown_distance"]

# validation data
df_X_validation = pull_values_by_index_from_on_ground(df_adsb_validation, cols_X)
df_validation = df_X_validation.merge(df_qar_validation, on="key")
df_X_validation = drop_cols_for_training(df_X_validation)
df_y_validation = df_validation["touchdown_distance"]

# test data: inference only
df_X_test = pull_values_by_index_from_on_ground(df_adsb_test, cols_X)
df_X_test = drop_cols_for_training(df_X_test)

# ## Compare several models:
reg = LazyRegressor(
    verbose=1, ignore_warnings=False, custom_metric=None, random_state=2023
)
models, preds = reg.fit(df_X_train, df_X_validation, df_y_train, df_y_validation)
models.sort_values(["RMSE"])
assert models.sort_values(["RMSE"]).index[0] == "ExtraTreesRegressor"


# ## Train RandomForestRegressor manually and compare with LazyRegressor result


def impute(df_train: pd.DataFrame, cols: List) -> pd.DataFrame:
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    index = df_train.index
    df = pd.DataFrame(imputer.fit_transform(df_train), columns=cols, index=index)
    return df


cols = df_X_train.columns
df_X_train_imputed = impute(df_X_train, cols=cols)
df_X_validation_imputed = impute(df_X_validation, cols=cols)

etree = ExtraTreesRegressor().fit(df_X_train_imputed, df_y_train)
etree.score(df_X_validation_imputed, df_y_validation)

# ## Submission on test data:
df_X_test_imputed = impute(df_X_test, cols=cols)
preds = etree.predict(df_X_test_imputed)

df_submission = pd.DataFrame(
    {"key": df_X_test_imputed.index, "touchdown_distance": preds, "exclude": 0}
)


def save_submission_csv(df: pd.DataFrame):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(dst_path.joinpath(f"ahmad{formatted_time}_3327735.csv"), index=False)


save_submission_csv(df_submission)
