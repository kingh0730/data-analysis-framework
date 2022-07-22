# %%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px


# %%
read = pd.read_csv("./temp/abalone.csv")
print(read)
df = read[["WholeWeight", "ShellWeight"]]
scaled = (df - df.mean()) / df.std()
scaled


# %%
px.scatter(scaled, x="WholeWeight", y="ShellWeight")


# %%
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)
pca_result


# %%
new_df = pd.DataFrame(pca_result)
new_df


# %%
px.scatter(new_df, x=0, y=1)


# %%
