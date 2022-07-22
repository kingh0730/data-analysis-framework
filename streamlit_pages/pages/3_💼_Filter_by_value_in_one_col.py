import streamlit as st
from utils import DAAS_INDEX_LIST, PRESET_FEATURES, django_setup


from base.models import (
    DataFrameFile,
    FilterByValueInOneCol,
    Job,
)


st.write(
    """
# Filter by value in one col
"""
)


with st.sidebar:
    input_df = st.selectbox(
        "Samples and features",
        DataFrameFile.objects.all(),
    )
    min_value = st.number_input(
        "min value",
    )
    max_value = st.number_input(
        "max value",
    )
    feature = st.selectbox(
        "Feature",
        zip(DAAS_INDEX_LIST.index, DAAS_INDEX_LIST["index_name"]),
    )


feature_index, feature_name = feature[0], feature[1]


with st.echo():

    def main() -> Job:
        result = FilterByValueInOneCol.objects.create(
            input=input_df,
            use_col=str(feature_index),
            min_value=min_value,
            max_value=max_value,
        )

        return result


if st.button("Execute main()"):
    st.info("Executing...")
    st.warning("Check terminal.")
    st.success(f"Success: {main()}")
