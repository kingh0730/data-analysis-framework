# %%
import pandas as pd
import numpy as np


# %%
df = pd.read_csv("./temp/186---FillNA___Tue_Jun_28_15-05-37_2022.csv", index_col=0)
df


# %%
def calc_relative_positions(series: pd.Series) -> pd.Series:
    min_value = min(series)
    max_value = max(series)
    dist = max_value - min_value
    return (series - min_value) / dist


df.apply(calc_relative_positions)


# %%
max(df["gov_id"])


# %%
def sort_series_and_calc_ordinal_numbers(series: pd.Series) -> pd.Series:
    """
    Parameters:
        series: should not have NaN values.
    """
    sorted_series = series.sort_values(ascending=False).reset_index(name="value")
    drop_dup = sorted_series.drop_duplicates(subset=["value"])
    dropped = pd.concat([drop_dup, sorted_series]).drop_duplicates(keep=False)

    def reassign_index(row: pd.Series) -> pd.Series:
        row["value"] = drop_dup.loc[drop_dup["value"] == row["value"]].index[0]
        return row

    set_good_index = dropped.apply(reassign_index, axis=1)
    drop_dup["value"] = drop_dup.index
    together = pd.concat([drop_dup, set_good_index]).set_index("index")["value"]

    return together


# %%
series = df["42367"]
series


# %%
ordinal_numbers = df.apply(sort_series_and_calc_ordinal_numbers)
ordinal_numbers


# %%
ordinal_numbers["42367"][1776]


# %% [markdown]
# Below is copying inside of above function


# %%
sorted_series = series.sort_values(ascending=False).reset_index(name="value")
# print(sorted_series)


drop_dup = sorted_series.drop_duplicates(subset=["value"])
# print(drop_dup)


dropped = pd.concat([drop_dup, sorted_series]).drop_duplicates(keep=False)
# print(dropped)


def reassign_index(row: pd.Series) -> pd.Series:
    row["value"] = drop_dup.loc[drop_dup["value"] == row["value"]].index[0]
    return row


set_good_index = dropped.apply(reassign_index, axis=1)
# print(set_good_index)


drop_dup["value"] = drop_dup.index
# print(drop_dup)


together = pd.concat([drop_dup, set_good_index]).set_index("index")["value"]
print(together)


# %%
print(series)
print(series[2012])
