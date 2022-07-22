# %%
import pandas as pd


# %%
input_df = pd.read_csv(
    "./temp/58818---AssociationRules___Thu_Jul__7_18-40-55_2022.csv", index_col=0
)
input_df


# %%
s: str = input_df["antecedents"][0]
s


# %%
s.strip(r"abcdefghijklmnopqrstuvwxyz(){}")


# %%
def deal_with_ante_and_cons(frozen_set: str) -> list[str]:
    stripped = frozen_set.strip(r"abcdefghijklmnopqrstuvwxyz(){}")
    splitted = [item.strip("'") for item in stripped.split(", ")]
    return splitted


ante_and_cons = input_df[["antecedents", "consequents"]]
dealt = ante_and_cons.apply(lambda series: series.apply(deal_with_ante_and_cons))
dealt


# %%
dealt["antecedents"][357]


# %%
