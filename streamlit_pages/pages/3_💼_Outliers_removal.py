import streamlit as st
from utils import django_setup


from base.models import (
    DataFrameFile,
    EachColOutliersIdentification,
    Job,
)


st.write(
    """
# Outliers removal
"""
)


with st.sidebar:
    input_df = st.selectbox(
        "Samples and features",
        DataFrameFile.objects.all(),
    )
    deviation_threshold = st.number_input(
        "deviation threshold",
        min_value=0,
    )


with st.echo():

    def main() -> Job:
        result = EachColOutliersIdentification.objects.create(
            input=input_df, deviation_threshold=deviation_threshold
        )

        return result


if st.button("Execute main()"):
    st.info("Executing...")
    st.warning("Check terminal.")
    st.success(f"Success: {main()}")
