# %%
import numpy as np
import pandas as pd


# %%
sorted_corr = pd.read_csv(
    "./temp/15---SortValuesInFeaturesPairwise___Thu_Jun_16_10-44-1_M2QQcDU.csv",
    index_col=[0, 1],
)
sorted_corr


# %%
filtered_corr = (
    sorted_corr.where(lambda x: abs(x) > 0.5).where(lambda x: abs(x) < 2).dropna()
)
filtered_corr


# %%
filtered_corr.to_csv(
    "./temp/filtered_corr_from_15---SortValuesInFeaturesPairwise___Thu_Jun_16_10-44-1_M2QQcDU.csv"
)


# %%
qcd = pd.read_csv(
    "./temp/13---DispersionCalculation___Thu_Jun_16_10-41-14_2022_I6jKfWo.csv",
    index_col=[0, 1],
)
qcd


# %%
filtered_corr["qcd1"]