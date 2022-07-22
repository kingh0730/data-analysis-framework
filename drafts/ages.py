# %%
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import remove_columns_that_have_too_many_nas


# %%
districts = pd.read_csv(
    "./temp/5---GovLevelFiltering___Wed_Jun_15_15-26-10_2022.csv",
    index_col=0,
)
districts = remove_columns_that_have_too_many_nas(districts, len(districts) * 0.75)
districts


# # %%
# pops = pd.DataFrame(
#     {
#         60: districts["56098"],
#         70: districts["56097"],
#         80: districts["56096"],
#         90: districts["59081"],
#         "pop": districts["65583"],
#     }
# )
# pops


# # %%
# pops["60_ratio"] = pops[60] / pops["pop"]
# pops["70_ratio"] = pops[70] / pops["pop"]
# pops["80_ratio"] = pops[80] / pops["pop"]
# pops["90_ratio"] = pops[90] / pops["pop"]
# pops


# %%
pops_year = pd.DataFrame(
    {
        "90_count": districts["154517"],
        "90_ratio": districts["154518"],
        "80_count": districts["154519"],
        "80_ratio": districts["154520"],
        "70_count": districts["154521"],
        "70_ratio": districts["154522"],
        "60_count": districts["154523"],
        "60_ratio": districts["154524"],
    }
)
pops_year


# %%
corrs = districts.corr(method="pearson")
corrs


# %%
sorted_corr = (
    corrs.where(np.triu(np.ones(corrs.shape), k=1).astype(bool))
    .unstack()
    .dropna()
    .sort_values(ascending=False)
)
sorted_corr


# %%
related_to_90_ratio = sorted_corr.filter(like="154518")
related_to_90_ratio


# %%
related_to_60_ratio = sorted_corr.filter(like="154524")
related_to_60_ratio


# %%
NINETIES = "154518"
SIXTIES = "154524"
# 农、林、牧、渔业企业数量占比（大类）
FARMING = "70612"

renamed = districts.rename(
    columns={
        NINETIES: f"{NINETIES}: 90 people ratio",
        SIXTIES: f"{SIXTIES}: 60 people ratio",
        FARMING: f"{FARMING}: Farming ratio",
    }
)
renamed.plot.scatter(
    f"{NINETIES}: 90 people ratio",
    f"{SIXTIES}: 60 people ratio",
)
renamed.plot.scatter(
    f"{SIXTIES}: 60 people ratio",
    f"{FARMING}: Farming ratio",
)
renamed.plot.scatter(
    f"{NINETIES}: 90 people ratio",
    f"{FARMING}: Farming ratio",
)


# %%
# 80后
districts.plot.scatter(NINETIES, "56095")


# %%
# 70后
districts.plot.scatter(NINETIES, "154522")


# %%
# 60
# 劳动力人口占常住人口的比重（月度）
districts.plot.scatter(SIXTIES, "391")


# %%
# 60
# 节能环保产业活跃度指数
districts.plot.scatter(SIXTIES, "118")


# %%
# 60
# 老年人口占常住人口的比重（年度）
districts.plot.scatter(SIXTIES, "154516")


# %%
expanded = related_to_90_ratio.to_frame().reset_index()
expanded


# %%
level_0 = related_to_90_ratio.groupby(level=0).count()
level_0


# %%
level_1 = related_to_90_ratio.groupby(level=1).count()
level_1


# %%
level_0.add(level_1, fill_value=0).sort_values()


# %%
