import pandas as pd
import streamlit as st
import plotly.express as px
from utils import (
    DAAS_INDEX_LIST,
    GOVS_INFO,
    MACRO_DATA_DIR_INDEX_NODES,
    month_int_to_str,
    django_setup,
)


from base import models


st.write(
    """
# Lines across time macro data
"""
)


# Inputs


with st.sidebar:
    dff = st.selectbox("Macro data", models.DataFrameFile.objects.all())
    feature = st.selectbox(
        "Feature",
        zip(MACRO_DATA_DIR_INDEX_NODES.index, MACRO_DATA_DIR_INDEX_NODES["node_name"]),
    )
    govs = st.multiselect("Govs", GOVS_INFO["name"])


feature_index, feature_name = str(feature[0]), feature[1]
feature_dict = {feature_index: feature_name}
govs_ids = [int(GOVS_INFO[GOVS_INFO["name"] == gov].index[0]) for gov in govs]
govs_dict = {govs_ids[i]: gov for i, gov in enumerate(govs)}


# Outputs


st.write("Feature:", feature_dict)
st.write("Govs:", govs_dict)


if govs:
    data_frame = pd.read_csv(
        dff.cached_file.path, usecols=["gov_id", "month", feature_index]
    )

    formatted = data_frame.rename(
        columns=feature_dict,
    )
    formatted["gov_name"] = [
        GOVS_INFO["name"][gov_id] for gov_id in formatted["gov_id"]
    ]
    formatted["month_str"] = [
        month_int_to_str(int(month_int)) for month_int in formatted["month"]
    ]

    line = px.line(
        formatted, x="month_str", y=feature_name, color="gov_name", markers=True
    )
    st.plotly_chart(line)

    st.write(data_frame)
