import streamlit as st
from utils import MeasureOfDispersion, django_setup


from base.models import (
    AppendInfoToFeaturesPairwiseFlat,
    AppendMacroDataInfoToFeaturesPairwiseFlat,
    AssociationRules,
    CalcValueOntologies,
    CorrelationCalculation,
    DataFrameFile,
    DiscretizeEachCol,
    DispersionCalculation,
    DownloadOneGovAcrossTime,
    DownloadOneGovMacroDataAcrossTime,
    DynamicTimeWarping,
    EachColOutliersIdentification,
    EachColPerCapita,
    FillNA,
    FilterValuesInFeaturesPairwiseFlat,
    Job,
    LowQualityRowsAndColsRemoval,
    Month,
    NormalizeEachCol,
    OneMonthGovLevelAndIndexes,
    SortValuesInFeaturesPairwise,
)


st.write(
    """
# Dynamic time warping macro data
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


with st.echo():

    def main() -> Job:
        # Download
        in1 = DownloadOneGovMacroDataAcrossTime.objects.create_download_one_gov_macro_data_across_time(
            gov_id=gov_id,
            months=months,
        )

        # Calculation
        normalized = NormalizeEachCol.objects.create(
            input=in1.output,
        )
        drop_cols = LowQualityRowsAndColsRemoval.objects.create(
            input=normalized.output,
            in_memory_inputs={
                "input": normalized.in_memory_outputs["output"],
            },
            min_quality_for_col=0.9,
            min_quality_for_row=0,
        )
        dtw = DynamicTimeWarping.objects.create(
            input=drop_cols.output,
            in_memory_inputs={
                "input": drop_cols.in_memory_outputs["output"],
            },
        )
        sorted_corr = SortValuesInFeaturesPairwise.objects.create(
            input=dtw.output,
        )
        filtered_corr = sorted_corr

        # Append info
        dispersion = DispersionCalculation.objects.create(
            input=in1.output,
            measure_of_dispersion=MeasureOfDispersion.CV2,
        )
        result = AppendMacroDataInfoToFeaturesPairwiseFlat.objects.create(
            input=filtered_corr.output,
            input_dispersion=dispersion.output,
        )

        return result


if st.button("Execute main()"):
    st.info("Executing...")
    st.warning("Check terminal.")
    st.success(f"Success: {main()}")
