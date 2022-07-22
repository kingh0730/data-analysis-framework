# %%
import pandas as pd
import numpy as np


# %%
def remove_outliers_series_with_median_deviation(
    series: pd.Series, outlier_sd_threshold: float
) -> pd.Series:
    dropped_na = series.dropna()
    deviation = np.abs(dropped_na - np.median(dropped_na))
    median_deviation = np.median(deviation)
    scaled_deviation = (deviation / median_deviation) if median_deviation else None

    result = (
        dropped_na[scaled_deviation < outlier_sd_threshold]
        if scaled_deviation is not None
        else pd.Series(index=series.index, dtype=np.float64)
    )

    return result


# %%
df = pd.read_csv("./temp/2018-12Wed_Jun__8_15-04-59_2022.csv", index_col=0)
df


# %%
remove_outliers_series_with_median_deviation(df["month"], 3)


# %%
same1 = df.apply(lambda s: remove_outliers_series_with_median_deviation(s, 3))


# %%
input_df = df.copy(deep=True)
for i, col in enumerate(input_df.columns):
    print(f"Progress: {i + 1} / {len(input_df.columns)}")
    input_df[col] = remove_outliers_series_with_median_deviation(input_df[col], 3)


# %%
same1.equals(input_df)


# %%
