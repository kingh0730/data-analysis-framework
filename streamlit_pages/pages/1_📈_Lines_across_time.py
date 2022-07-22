import pandas as pd
import streamlit as st
import plotly.express as px
from utils import DAAS_INDEX_LIST, GOVS_INFO, month_int_to_str, django_setup


from base import models


st.write(
    """
# Lines across time
"""
)


# Inputs


with st.sidebar:
    feature = st.selectbox(
        "Feature",
        zip(DAAS_INDEX_LIST.index, DAAS_INDEX_LIST["index_name"]),
    )
    govs = st.multiselect("Govs", GOVS_INFO["name"])
    months = st.multiselect("Months", models.OneMonthGovsAndIndexes.objects.all())


feature_index, feature_name = str(feature[0]), feature[1]
feature_dict = {feature_index: feature_name}
govs_ids = [int(GOVS_INFO[GOVS_INFO["name"] == gov].index[0]) for gov in govs]
govs_dict = {govs_ids[i]: gov for i, gov in enumerate(govs)}


# Outputs


st.write("Feature:", feature_dict)
st.write("Govs:", govs_dict)


if months:
    list_data_frames: list[pd.DataFrame] = []
    for month in months:
        read_data_frame = pd.read_csv(
            month.cached_file.path, usecols=["gov_id", "month", feature_index]
        )
        this_data_frame = read_data_frame.loc[
            read_data_frame["gov_id"].isin(govs_ids), ["gov_id", "month", feature_index]
        ]
        list_data_frames.append(this_data_frame)
    data_frame = pd.concat(list_data_frames)

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
