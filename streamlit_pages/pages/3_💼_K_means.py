import json
import streamlit as st
from utils import DAAS_INDEX_LIST, PRESET_FEATURES, django_setup


from base.models import (
    DataFrameFile,
    Job,
    KMeans,
)


st.write(
    """
# K means
"""
)


with st.sidebar:
    input_df = st.selectbox(
        "Samples and features",
        DataFrameFile.objects.all(),
    )
    n_clusters = st.number_input(
        "number of clusters",
        min_value=1,
    )
    use_all_cols = st.radio(
        "Use all cols?",
        [True, False],
    )
    preset = st.selectbox(
        "Preset features",
        PRESET_FEATURES.keys(),
        disabled=use_all_cols,
    )
    features = st.multiselect(
        "Features",
        zip(DAAS_INDEX_LIST.index, DAAS_INDEX_LIST["index_name"]),
        disabled=bool(preset) or use_all_cols,
    )


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


with st.echo():

    def main() -> Job:
        result = KMeans.objects.create(
            input=input_df,
            n_clusters=n_clusters,
            use_cols="" if use_all_cols else json.dumps(list(features_dict.keys())),
        )

        return result


if st.button("Execute main()"):
    st.info("Executing...")
    st.warning("Check terminal.")
    st.success(f"Success: {main()}")
