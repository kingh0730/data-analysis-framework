# %%
import pandas as pd


# %%
series = pd.Series(
    [6, 34, 234, 5, 0, -234, 3, -44, -4, 3245, 2, 7, 88, 6, 5, 4, -4], name="value"
)


# %%
sorted = series.sort_values(ascending=False).reset_index(name="value")
sorted


# %%
drop_dup = sorted.drop_duplicates(subset=["value"])
drop_dup


# %%
dropped = pd.concat([drop_dup, sorted]).drop_duplicates(keep=False)
dropped


# %%
def reassign_index(row: pd.Series) -> pd.Series:
    # print(f"{row=}")
    # print(f"{type(row)=}")
    # print(f"{row['index']=}")
    # print(f"{row.name=}")
    # print(f"{sorted.loc[row.name]=}")
    # print(f"{sorted.loc[row.name, 'index']=}")
    # print(f"{drop_dup.loc[drop_dup['value'] == row['value']]=}")
    # print(f"{drop_dup.loc[drop_dup['value'] == row['value'], 'index']=}")
    print(f"{drop_dup.loc[drop_dup['value'] == row['value']].index[0]=}")
    row["index"] = drop_dup.loc[drop_dup["value"] == row["value"]].index[0]
    return row


set_good_index = dropped.apply(reassign_index, axis=1).set_index("index")
set_good_index


# %%
pd.concat([drop_dup, set_good_index]).sort_index()

# %%
