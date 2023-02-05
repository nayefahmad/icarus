# # EDA for landing distance modeling competition

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pdpipe as pdp

data_path = Path(r"C:\Nayef\icarus\data")

df_adsb_train = pd.read_csv(data_path.joinpath("adsb_train.csv"))
df_qar_train = pd.read_csv(data_path.joinpath("qar_train.csv"))
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
x.sort_values().plot.hist()
plt.show()

x1 = df_join.query("on_ground == 1")
x1 = x1.groupby("key").size()
assert min(x1) == 3
x1.sort_values().plot.hist()
plt.show()

x2 = df_join.query("on_ground == 0")
x2 = x2.groupby("key").size()
assert min(x2) == 5
x2.sort_values().plot.hist()
plt.show()

result = {}
for key in df_adsb_train["key"].unique():
    # do stuff
    result["key"] = key

    cols = [
        "distance_from_threshold_minus1",
        "distance_from_threshold_minus2",
        "distance_from_threshold_0",
        "distance_from_threshold_plus1",
    ]

    for col in cols:
        result[col] = "val"

pd.DataFrame(result, index=range(1500))
