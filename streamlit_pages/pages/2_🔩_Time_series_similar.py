import streamlit as st
from utils import MeasureOfDispersion, django_setup


from base.models import (
    AppendInfoToFeaturesPairwiseFlat,
    CalcValueOntologies,
    ConcatDataFrames,
    CorrelationCalculation,
    DataFrameFile,
    DispersionCalculation,
    DownloadOneGovAcrossTime,
    EachColOutliersIdentification,
    EachColPerCapita,
    FilterByInfo,
    FilterValuesInFeaturesPairwiseFlat,
    Job,
    LowQualityRowsAndColsRemoval,
    Month,
    OneMonthGovLevelAndIndexes,
    SortValuesInFeaturesPairwise,
)


st.write(
    """
# Time series similarities
"""
)


with st.sidebar:
    gov_id = st.selectbox(
        "gov_id",
        range(10),
    )
    months = st.multiselect(
        "Months",
        Month.objects.all(),
    )
    months_future = st.multiselect(
        "Months future",
        Month.objects.all(),
    )
    method = st.radio(
        "method",
        list(map(lambda pair: pair[1], CorrelationCalculation.METHODS_OF_CORRELATION)),
    )


with st.echo():

    def main() -> Job:
        # Download
        in1 = DownloadOneGovAcrossTime.objects.create_download_one_gov_across_time(
            gov_id=gov_id,
            months=months,
        )
        in2 = DownloadOneGovAcrossTime.objects.create_download_one_gov_across_time(
            gov_id=gov_id,
            months=months_future,
        )
        combined = ConcatDataFrames.objects.create(
            input1=in1.output,
            input2=in2.output,
            comment="future",
        )

        # Calculation
        outliers_identification = EachColOutliersIdentification.objects.create(
            input=combined.output,
            deviation_threshold=5,
        )
        drop_cols = LowQualityRowsAndColsRemoval.objects.create(
            input=outliers_identification.output,
            min_quality_for_col=0.75,
            min_quality_for_row=0,
        )
        # drop_rows = LowQualityRowsAndColsRemoval.objects.create(
        #     input=drop_cols.output,
        #     min_quality_for_col=0,
        #     min_quality_for_row=0.9,
        # )
        drop_rows = drop_cols
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
        filtered = FilterByInfo.objects.create(
            input=result.output,
            no_same_type=True,
            no_year_period=True,
            # min_dispersion=0.5,
        )

        return filtered


if st.button("Execute main()"):
    st.info("Executing...")
    st.warning("Check terminal.")
    st.success(f"Success: {main()}")
