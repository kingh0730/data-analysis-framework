# %%
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth


# %%
input_df = pd.read_csv(
    "./temp/32861---DiscretizeEachCol___Mon_Jul__4_16-32-57_2022.csv", index_col=0
)
input_df


# %% Fill nan with 0
filled_na = input_df.fillna(0)
filled_na


# %%
all_dummies = []
filled_na.apply(
    lambda series: all_dummies.append(
        pd.get_dummies(series).rename(
            columns={
                0: f"{series.name}---0",
                1: f"{series.name}---1",
            }
        )
    )
)
print(len(all_dummies))


# %%
one_hot = pd.DataFrame(index=filled_na.index)
for dummy in all_dummies[:5]:
    one_hot = one_hot.append(dummy)
pd.concat(all_dummies, axis=1)


# %%
MIN_SUPPORT = 0.4
try:
    apriori_results = apriori(
        filled_na,
        use_colnames=True,
        min_support=MIN_SUPPORT,
        max_len=1,
        # low_memory=False,
        # verbose=1,
    )
except MemoryError as exception:
    print(exception)
    apriori_results = fpgrowth(
        filled_na,
        use_colnames=True,
        min_support=MIN_SUPPORT,
        max_len=1,
        # low_memory=True,
        # verbose=1,
    )

apriori_results


# %%
df_ar = association_rules(apriori_results, metric="confidence", min_threshold=0.6)
df_ar


# %%
sum(df_ar["confidence"])


# %%
fpgrowth(
    filled_na,
    use_colnames=True,
    min_support=MIN_SUPPORT,
    max_len=3,
    # low_memory=True,
    # verbose=1,
)
