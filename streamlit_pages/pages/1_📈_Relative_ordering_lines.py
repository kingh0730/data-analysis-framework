import pandas as pd
import streamlit as st
import plotly.express as px
from utils import DAAS_INDEX_LIST, GOVS_INFO, PRESET_FEATURES, django_setup


from base import models


st.write(
    """
# Relative ordering lines
"""
)


# Inputs


with st.sidebar:
    dff = st.selectbox("Relative ordering", models.DataFrameFile.objects.all())
    preset = st.selectbox("Preset features", PRESET_FEATURES.keys())
    features = st.multiselect(
        "Features",
        zip(DAAS_INDEX_LIST.index, DAAS_INDEX_LIST["index_name"]),
        disabled=bool(preset),
    )
    show_neg_gov_id = st.checkbox("Show (neg) gov id?")


selected_features_indexes = [f[0] for f in features]
selected_features_names = [f[1] for f in features]


features_indexes = PRESET_FEATURES[preset] if preset else selected_features_indexes
features_names = (
    [DAAS_INDEX_LIST["index_name"][i] for i in PRESET_FEATURES[preset]]
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
    formatted["neg gov_id"] = -formatted["gov_id"]

    plot_parallel_coordinates = px.parallel_coordinates(
        formatted[["neg gov_id"] + features_names],
        dimensions=None if show_neg_gov_id else features_names,
        color="neg gov_id" if show_neg_gov_id else features_names[0],
    )
    st.plotly_chart(plot_parallel_coordinates)
