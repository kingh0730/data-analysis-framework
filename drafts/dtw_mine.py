# %%
import numpy as np
import pandas as pd
from dtw import dtw


# %%
df = pd.read_csv(
    "./temp/9851---DownloadOneGovMacroDataAcrossTime___Fri_Jul__1_17-16-42_2022.csv",
    index_col=0,
)
df


# %%
manhattan_distance = lambda x, y: np.abs(x - y)
dtw_distance, *_ = dtw(df["1"], df["2"], manhattan_distance)
dtw_distance


# %%
keep_track: dict[pd.Series, dict[pd.Series, float]] = {}


def dtw_wrapper(
    series: pd.Series,
    series2: pd.Series,
) -> float:
    if series2.name in keep_track and series.name in keep_track[series2.name]:
        return keep_track[series2.name][series.name]
    result = dtw(series, series2, manhattan_distance)[0]
    keep_track[series.name] = (
        keep_track[series.name] if series.name in keep_track else {}
    )
    keep_track[series.name][series2.name] = result
    return result


df.apply(
    lambda series: pd.Series(
        [dtw_wrapper(series, series2) for _, series2 in df.iteritems()],
        df.columns,
    )
)
