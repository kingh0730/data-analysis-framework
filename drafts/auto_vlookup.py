# %%
import pandas as pd
from utils import DAAS_INDEX_LIST
import my_package as mp


# %%
pairs = pd.read_csv(
    "./drafts/temp/15---SortValuesInFeaturesPairwise___Thu_Jun_16_10-44-1_M2QQcDU.csv",
    names=["index1", "index2", "value"],
    header=0,
)
print(pairs)
drop_strings = pairs.apply(pd.to_numeric, errors="coerce")
pairs = drop_strings.dropna()
pairs


# %%
merged1 = pd.merge(
    pairs,
    DAAS_INDEX_LIST[
        [
            "index_name",
            "year_period",
            "type",
            "sub_type",
        ]
    ],
    left_on="index1",
    right_on="index_id",
).rename(
    columns={
        "index_name": "index_name1",
        "year_period": "year_period1",
        "type": "type1",
        "sub_type": "sub_type1",
    }
)
merged1


# %%
DAAS_INDEX_LIST


# %%
input_df = pd.read_csv(
    "./drafts/temp/15---SortValuesInFeaturesPairwise___Thu_Jun_16_10-44-1_M2QQcDU.csv",
    names=[1, 2, "value"],
    header=0,
)
print(input_df)
drop_strings = input_df.apply(pd.to_numeric, errors="coerce")
input_df = drop_strings.dropna()


# Constants
KEEP_THESE = [
    "index_name",
    "year_period",
    "type",
    "sub_type",
]


# Per capita
def merge_once(left: pd.DataFrame, index_num: int) -> pd.DataFrame:
    return pd.merge(
        left,
        DAAS_INDEX_LIST[KEEP_THESE],
        left_on=index_num,
        right_on="index_id",
    ).rename(columns={k: f"{k}{index_num}" for k in KEEP_THESE})


merged_once1 = merge_once(input_df, 1)
merged_once1


# %%
merge_once(merged_once1, 2)


# %%
