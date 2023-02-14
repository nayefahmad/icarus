# # EDA for landing distance modeling competition

from itertools import chain
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import pdpipe as pdp
from lazypredict.Supervised import LazyRegressor

data_path = Path(r"C:\Nayef\icarus\data")

df_adsb_train = pd.read_csv(data_path.joinpath("adsb_train.csv"))  # todo: split this
df_adsb_test = pd.read_csv(
    data_path.joinpath("adsb_test.csv")
)  # todo: use only for inference

df_qar_train = pd.read_csv(data_path.joinpath("qar_train.csv"))
df_qar_test = pd.read_csv(data_path.joinpath("qar_test.csv"))  # todo: doesn't exist

df_altitude_train = pd.read_csv(data_path.joinpath("train_altitude_at_landing.csv"))

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
df_adsb_test = pipe_adsb(df_adsb_test)

pipe_qar = pdp.PdPipeline([pdp.ColumnDtypeEnforcer({"key": pd.StringDtype()})])
df_qar_train = pipe_qar(df_qar_train)
df_qar_train.info()
df_qar_train.describe(include="all").T

pipe_altitude = pdp.PdPipeline([pdp.ColumnDtypeEnforcer({"key": pd.StringDtype()})])
df_altitude_train = pipe_altitude(df_altitude_train)
df_altitude_train.info()
df_altitude_train.describe(include="all").T


# ## Join train datasets

df_join = df_adsb_train.merge(df_qar_train, on="key").merge(df_altitude_train, on="key")
df_join.info()

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


# train data
df_X_train = pull_values_by_index_from_on_ground(df_adsb_train, cols_X)
df_train = df_X_train.merge(df_qar_train, on="key")
df_y_train = df_train["touchdown_distance"]

# test data
df_X_test = pull_values_by_index_from_on_ground(df_adsb_test, cols_X)
df_test = df_X_test.merge(df_qar_test)
df_y_test = df_test["touchdown_distance"]


reg = LazyRegressor()
models, preds = reg.fit(df_X_train, df_X_test, df_y_train, df_y_test)
