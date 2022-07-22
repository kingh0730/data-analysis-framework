# %%
import json
import pandas as pd

from utils import (
    DAAS_INDEX_LIST,
    GovLevel,
    get_gov_level_ids,
    month_int_to_str,
    zk2861api,
    SECURE_KEY,
)


# %%
PROD_INDEXES = None
with open(r"./drafts/temp/PROD_INDEXES_A_PRIORI.json", "r") as f:
    PROD_INDEXES = json.load(f)
print(f"PROD_INDEXES length: {len(PROD_INDEXES)}")


# %%
def get_values_one_month(month: int, gov_ids: list, index_ids: list) -> pd.DataFrame:
    to_be_saved = pd.DataFrame(
        {"gov_id": pd.Series([], dtype="int"), "month": pd.Series([], dtype="int")}
    )

    for gov_id in gov_ids:
        row = {
            "gov_id": gov_id,
            "month": month,
        }

        api_ret = zk2861api.api_get_gov_indexs_data(
            SECURE_KEY,
            gov_id,
            index_ids,
            f"{month_int_to_str(month)}-01",
            timeout=60,
        )
        data = api_ret["data"]
        for col in data:
            row[col["index_id"]] = col["value"]

        to_be_saved = to_be_saved.append(row, ignore_index=True)

    return to_be_saved


# %%
for gov_id in get_gov_level_ids(GovLevel.ALL):
    print(gov_id)
    api_ret = zk2861api.api_get_gov_indexs_data(
        SECURE_KEY,
        gov_id,
        DAAS_INDEX_LIST.index.to_list(),
        f"{month_int_to_str(252)}-01",
        timeout=60,
    )
    pd.DataFrame(api_ret["data"])


# %%
