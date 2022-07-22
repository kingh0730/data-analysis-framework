import os
import traceback
import streamlit as st
from utils import GovLevel, MeasureOfDispersion, django_setup, get_gov_level_ids


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
    RemoveDispersionSmallCol,
    SortValuesInFeaturesPairwise,
)


st.write(
    """
# Time series similarities
"""
)


with st.sidebar:
    months = st.multiselect(
        "Months",
        Month.objects.all(),
    )
    method = st.radio(
        "method",
        list(map(lambda pair: pair[1], CorrelationCalculation.METHODS_OF_CORRELATION)),
    )


with st.echo():

    def main() -> list[Job]:
        def do_once(gov_id_once: int) -> Job:
            # Download
            in1 = DownloadOneGovAcrossTime.objects.create_download_one_gov_across_time(
                gov_id=gov_id_once,
                months=months,
            )

            # Calculation
            # outliers_identification = EachColOutliersIdentification.objects.create(
            #     input=in1.output,
            #     deviation_threshold=6,
            #     should_output_in_file=False,
            # )
            outliers_identification = in1
            drop_cols = LowQualityRowsAndColsRemoval.objects.create(
                input=outliers_identification.output,
                # in_memory_inputs={
                #     "input": outliers_identification.in_memory_outputs["output"],
                # },
                min_quality_for_col=0.8,
                min_quality_for_row=0,
                should_output_in_file=False,
            )
            remove_dispersion_small_col = RemoveDispersionSmallCol.objects.create(
                input=drop_cols.output,
                in_memory_inputs={
                    "input": drop_cols.in_memory_outputs["output"],
                },
                measure_of_dispersion=MeasureOfDispersion.QCD,
                min_dispersion=0.3,
            )
            corr = CorrelationCalculation.objects.create(
                input=remove_dispersion_small_col.output,
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
                input=remove_dispersion_small_col.output,
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
                min_dispersion=0.3,
            )

            # Remove cached files
            os.remove(in1.output.cached_file.path)
            # os.remove(remove_dispersion_small_col.output.cached_file.path)
            os.remove(corr.output.cached_file.path)
            os.remove(sorted_corr.output.cached_file.path)
            os.remove(filtered_corr.output.cached_file.path)
            os.remove(value_ontologies.output.cached_file.path)
            os.remove(dispersion.output.cached_file.path)
            os.remove(result.output.cached_file.path)
            os.rename(
                filtered.output.cached_file.path,
                f"./out/micro_data_quarter_time_series/{gov_id_once}.csv",
            )

            return filtered

        result = []
        for gov_id_once in get_gov_level_ids(GovLevel.ALL):
            try:
                result_once = do_once(gov_id_once)
                result.append(result_once)
            except Exception as exception:
                st.error(f"{gov_id_once}: {exception}")
                # st.error(traceback.format_exc())

        return result


if st.button("Execute main()"):
    st.info("Executing...")
    st.warning("Check terminal.")
    st.success(f"Success: {main()}")
