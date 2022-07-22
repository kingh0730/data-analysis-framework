# %%
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
districts = pd.read_csv(
    "./temp/18---LowQualityColsRemoval___Thu_Jun_16_17-59-55_2022_VOhnpvc (1).csv",
    index_col=0,
)
districts


# %%
smaller = districts
squares = smaller.apply(np.square).rename(columns=lambda name: str(name) + "_2")
with_squares = pd.concat([smaller, squares], axis=1)
with_squares


# %%
corr = with_squares.corr(method="pearson")
corr


# %%
def sort_corr(corr_input: pd.DataFrame) -> pd.DataFrame:
    sorted_corr = (
        corr_input.where(np.triu(np.ones(corr_input.shape), k=1).astype(bool))
        .unstack()
        .dropna()
        .sort_values(ascending=False)
    )

    return sorted_corr


# %%
sorted = sort_corr(corr)
sorted = sorted.where(lambda x: abs(x) > 0.5).where(lambda x: abs(x) < 2).dropna()
sorted


# %%
sorted_df = pd.DataFrame(sorted).reset_index()
squares_and_not = sorted_df[
    (sorted_df["level_0"].str.contains("_2") & ~sorted_df["level_1"].str.contains("_2"))
    | (
        ~sorted_df["level_0"].str.contains("_2")
        & sorted_df["level_1"].str.contains("_2")
    )
]
squares_and_not


# %%
squares_and_not["level_0_ori"] = squares_and_not["level_0"].apply(lambda x: x[:-2])
squares_and_not


# %%
squares_and_not.to_csv("./temp/squares.csv")


# %%
