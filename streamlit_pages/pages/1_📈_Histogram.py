import pandas as pd
import streamlit as st
import plotly.express as px
from utils import (
    DAAS_INDEX_LIST,
    GOVS_INFO,
    GovLevel,
    django_setup,
    get_gov_level_ids,
    month_int_to_str,
)


from base import models


st.write(
    """
# Histogram
"""
)


# Inputs


with st.sidebar:
    dff = st.selectbox("Samples and features", models.DataFrameFile.objects.all())
    feature = st.selectbox(
        "Feature",
        zip(DAAS_INDEX_LIST.index, DAAS_INDEX_LIST["index_name"]),
    )
    preset_govs = st.selectbox("Preset govs", GovLevel)
    _govs = st.multiselect("Govs", GOVS_INFO["name"], disabled=bool(preset_govs))


feature_index, feature_name = str(feature[0]), feature[1]
feature_dict = {feature_index: feature_name}
gov_ids = get_gov_level_ids(preset_govs)


data_frame = pd.read_csv(dff.cached_file.path, index_col=0)
formatted = data_frame.rename(
    columns=feature_dict,
)
govs_only = formatted.loc[
    formatted["gov_id"].isin(gov_ids), ["gov_id", "month", feature_name]
]
govs_only["gov_name"] = [GOVS_INFO["name"][gov_id] for gov_id in govs_only["gov_id"]]
govs_only["month"] = [
    month_int_to_str(int(month_int)) for month_int in govs_only["month"]
]


# Draw


hist = px.histogram(
    govs_only, x=feature_name, marginal="rug", hover_data=govs_only.columns
)
st.plotly_chart(hist)


st.write(govs_only)
