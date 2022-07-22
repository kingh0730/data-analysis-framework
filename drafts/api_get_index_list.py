# %%
from utils import SECURE_KEY
from utils.zk2861api import api_get_index_list


res = api_get_index_list(SECURE_KEY, timeout=30)
res


# %%
all_dates = [item["new_date"] for item in res["data"]]
all_dates


# %%
print(len([d for d in all_dates if d[:4] == "2022"]))
print(len([d for d in all_dates if d[:4] == "2021"]))
print(len([d for d in all_dates if d[:4] == "2020"]))
print(len([d for d in all_dates if d[:4] == "2019"]))


# %%
