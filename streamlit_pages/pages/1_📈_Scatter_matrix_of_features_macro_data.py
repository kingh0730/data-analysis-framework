import pandas as pd
import streamlit as st
import plotly.express as px
from utils import (
    DAAS_INDEX_LIST,
    GOVS_INFO,
    MACRO_DATA_DIR_INDEX_NODES,
    django_setup,
)


from base import models


st.write(
    """
# Scatter matrix of features macro data
"""
)


# Inputs


PRESET_FEATURES_MACRO_DATA = {None: []}


with st.sidebar:
    dff = st.selectbox("Samples and features", models.DataFrameFile.objects.all())
    preset = st.selectbox("Preset features", PRESET_FEATURES_MACRO_DATA.keys())
    features = st.multiselect(
        "Features",
        zip(MACRO_DATA_DIR_INDEX_NODES.index, MACRO_DATA_DIR_INDEX_NODES["node_name"]),
        disabled=bool(preset),
    )


selected_features_indexes = [f[0] for f in features]
selected_features_names = [f[1] for f in features]


features_indexes = (
    PRESET_FEATURES_MACRO_DATA[preset] if preset else selected_features_indexes
)
features_names = (
    [
        MACRO_DATA_DIR_INDEX_NODES["node_name"][i]
        for i in PRESET_FEATURES_MACRO_DATA[preset]
    ]
    if preset
    else selected_features_names
)
features_dict = {
    str(i): features_names[position] for position, i in enumerate(features_indexes)
}


# Outputs


st.write("Features:", features_dict)


if len(features_indexes):
    data_frame = pd.read_csv(dff.cached_file.path, index_col=0)
    formatted = data_frame.rename(columns=features_dict)
    formatted["gov_name"] = [
        GOVS_INFO["name"][gov_id] for gov_id in formatted["gov_id"]
    ]

    plot_scatter_matrix = px.scatter_matrix(
        formatted, dimensions=features_names, hover_data=["gov_id", "gov_name"]
    )
    st.plotly_chart(plot_scatter_matrix)
