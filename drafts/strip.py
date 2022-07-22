# %%
import pandas as pd
import numpy as np


# %%
s1 = pd.Series(["1. Ant.   ---1234", "2. Bee!\n      ", "3. Cat?\t", np.nan])
s1


# %%
s2 = pd.Series(["a. 123 ", "a. 234_sad---again---asdfasdf", np.nan, "c.9384_dfdfdf"])
s2


# %%
s3 = pd.Series(["hi\t\t\t", "Hii", "123", "____---asdf"])
s3


# %%
df = pd.DataFrame({"s1": s1, "s2": s2, "s3": s3})
df


# %%
stripped = df[["s1", "s2"]].apply(lambda s: s.replace("---.*$", "", regex=True))
stripped


# %%
# df[["s1", "s2"]] = stripped
# df


# %%
df[["s1", "s2"]].apply(lambda s: s.replace("^.*---", "---", regex=True))


# %%
