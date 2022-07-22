# %%
import pandas as pd
import numpy as np


# %%
df = pd.read_csv("./temp/58784---FillNA___Thu_Jul__7_17-27-29_2022.csv", index_col=0)
df


# %%
def discretize_and_to_one_hot(
    input_df: pd.DataFrame, number_of_bins: int
) -> pd.DataFrame:
    discretized = input_df.apply(
        lambda s: pd.qcut(s, number_of_bins, labels=False, duplicates="drop")
    )
    # print(number_of_bins, discretized)
    print(len(discretized["65593"].where(lambda x: x == 0).dropna()))
    print(len(discretized["65593"].where(lambda x: x == 1).dropna()))
    print(len(discretized["65593"].where(lambda x: x == 2).dropna()))
    print(len(discretized["57284"].where(lambda x: x == 0).dropna()))
    print(len(discretized["57284"].where(lambda x: x == 1).dropna()))
    print(len(discretized["57284"].where(lambda x: x == 2).dropna()))

    # One hot
    def rename_columns(series: pd.Series) -> dict[int, str]:
        return {i: f"{series.name}---{i}" for i in range(number_of_bins)}

    all_dummies = []
    discretized.apply(
        lambda series: all_dummies.append(
            pd.get_dummies(series).rename(columns=rename_columns(series))
        )
    )

    one_hot = pd.concat(all_dummies, axis=1)

    return one_hot


# %%
discretize_and_to_one_hot(df, 3)


# %%
