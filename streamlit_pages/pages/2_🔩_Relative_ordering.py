import streamlit as st
from utils import django_setup


from base.models import (
    METRIC_KINDS,
    AppendInfoToFeaturesPairwiseFlat,
    DataFrameFile,
    DistanceBetweenColPairs,
    EachColRelativeOrdering,
    FillNA,
    Job,
    OneMonthGovLevelAndIndexes,
    SortValuesInFeaturesPairwise,
)


st.write(
    """
# Relative ordering
"""
)


with st.sidebar:
    input_df = st.selectbox(
        # "Data: one month, one gov level",
        # OneMonthGovLevelAndIndexes.objects.all(),
        "Samples and features",
        DataFrameFile.objects.all(),
    )
    metric_between_ordering = st.radio(
        "Metric between ordering",
        list(map(lambda pair: pair[1], METRIC_KINDS)),
    )


with st.echo():

    def main() -> Job:
        fill_na = FillNA.objects.create(
            input=input_df,
        )
        each_col_relative_ordering = EachColRelativeOrdering.objects.create(
            input=fill_na.output,
            method_of_relative_ordering=EachColRelativeOrdering.ORD,
        )
        dist_matrix = DistanceBetweenColPairs.objects.create(
            input=each_col_relative_ordering.output,
            metric=metric_between_ordering,
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
