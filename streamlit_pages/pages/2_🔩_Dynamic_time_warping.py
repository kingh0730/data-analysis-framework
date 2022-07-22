import streamlit as st
from utils import MeasureOfDispersion, django_setup


from base.models import (
    AppendInfoToFeaturesPairwiseFlat,
    AssociationRules,
    CalcValueOntologies,
    CorrelationCalculation,
    DataFrameFile,
    DiscretizeEachCol,
    DispersionCalculation,
    DownloadOneGovAcrossTime,
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
    RemoveDispersionSmallCol,
    SortValuesInFeaturesPairwise,
)


st.write(
    """
# Dynamic time warping
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
        in1 = DownloadOneGovAcrossTime.objects.create_download_one_gov_across_time(
            gov_id=gov_id,
            months=months,
        )

        # Calculation
        outliers_identification = EachColOutliersIdentification.objects.create(
            input=in1.output,
            deviation_threshold=6,
        )
        drop_cols = LowQualityRowsAndColsRemoval.objects.create(
            input=outliers_identification.output,
            in_memory_inputs={
                "input": outliers_identification.in_memory_outputs["output"],
            },
            min_quality_for_col=0.8,
            min_quality_for_row=0,
        )
        remove_dispersion_small_col = RemoveDispersionSmallCol.objects.create(
            input=drop_cols.output,
            in_memory_inputs={
                "input": drop_cols.in_memory_outputs["output"],
            },
            measure_of_dispersion=MeasureOfDispersion.CV2,
            min_dispersion=1,
        )
        fill_na = FillNA.objects.create(
            input=remove_dispersion_small_col.output,
        )
        normalized = NormalizeEachCol.objects.create(
            input=fill_na.output,
        )
        dtw = DynamicTimeWarping.objects.create(
            input=normalized.output,
        )
        sorted_corr = SortValuesInFeaturesPairwise.objects.create(
            input=dtw.output,
        )
        filtered_corr = sorted_corr

        # Append info
        value_ontologies = CalcValueOntologies.objects.create()
        dispersion = DispersionCalculation.objects.create(
            input=drop_cols.output,
            measure_of_dispersion=MeasureOfDispersion.CV2,
        )
        result = AppendInfoToFeaturesPairwiseFlat.objects.create(
            input=filtered_corr.output,
            input_value_ontologies=value_ontologies.output,
            input_dispersion=dispersion.output,
        )
        result = dtw

        return result


if st.button("Execute main()"):
    st.info("Executing...")
    st.warning("Check terminal.")
    st.success(f"Success: {main()}")
