import streamlit as st
from utils import django_setup


from base.models import (
    AppendInfoToFeaturesPairwiseFlat,
    Job,
    OneMonthGovsAndIndexes,
    SortValuesInFeaturesPairwise,
    TimeSeriesSimilarities,
)


st.write(
    """
# Time series similarities
"""
)


with st.sidebar:
    months = st.multiselect(
        "Months",
        OneMonthGovsAndIndexes.objects.all(),
    )


with st.echo():

    def main() -> Job:
        dist_matrix = TimeSeriesSimilarities.objects.create_time_series_similarities(
            months=months,
        )
        sorted_distances = SortValuesInFeaturesPairwise.objects.create(
            input=dist_matrix.output,
        )
        result = AppendInfoToFeaturesPairwiseFlat.objects.create(
            input=sorted_distances.output,
        )

        return result


if st.button("Execute main()"):
    st.info("Executing...")
    st.warning("Check terminal.")
    st.success(f"Success: {main()}")
