import streamlit as st
from utils import MeasureOfDispersion, django_setup


from base.models import (
    AppendInfoToFeaturesPairwiseFlat,
    CalcValueOntologies,
    CorrelationCalculation,
    DataFrameFile,
    DispersionCalculation,
    EachColOutliersIdentification,
    EachColPerCapita,
    FilterValuesInFeaturesPairwiseFlat,
    Job,
    LowQualityRowsAndColsRemoval,
    OneMonthGovLevelAndIndexes,
    SortValuesInFeaturesPairwise,
)


st.write(
    """
# Correlation
"""
)


with st.sidebar:
    input_df = st.selectbox(
        "Samples and features",
        DataFrameFile.objects.all(),
    )
    method = st.radio(
        "method",
        list(map(lambda pair: pair[1], CorrelationCalculation.METHODS_OF_CORRELATION)),
    )


with st.echo():

    def main() -> Job:
        # Calculation
        per_capita = EachColPerCapita.objects.create(
            input=input_df,
        )
        outliers_identification = EachColOutliersIdentification.objects.create(
            input=per_capita.output,
            deviation_threshold=5,
        )
        drop_cols = LowQualityRowsAndColsRemoval.objects.create(
            input=outliers_identification.output,
            min_quality_for_col=0.75,
            min_quality_for_row=0,
        )
        drop_rows = LowQualityRowsAndColsRemoval.objects.create(
            input=drop_cols.output,
            min_quality_for_col=0,
            min_quality_for_row=0.9,
        )
        corr = CorrelationCalculation.objects.create(
            input=drop_rows.output,
            method=method,
        )
        sorted_corr = SortValuesInFeaturesPairwise.objects.create(
            input=corr.output,
        )
        filtered_corr = FilterValuesInFeaturesPairwiseFlat.objects.create(
            input=sorted_corr.output,
            func_before_filter=FilterValuesInFeaturesPairwiseFlat.ABS,
            min_value=0.5,
        )

        # Append info
        value_ontologies = CalcValueOntologies.objects.create()
        dispersion = DispersionCalculation.objects.create(
            input=drop_rows.output,
            measure_of_dispersion=MeasureOfDispersion.QCD,
        )
        result = AppendInfoToFeaturesPairwiseFlat.objects.create(
            input=filtered_corr.output,
            input_value_ontologies=value_ontologies.output,
            input_dispersion=dispersion.output,
        )

        return result


if st.button("Execute main()"):
    st.info("Executing...")
    st.warning("Check terminal.")
    st.success(f"Success: {main()}")
