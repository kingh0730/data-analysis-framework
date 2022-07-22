import os
import traceback
import streamlit as st
from utils import GovLevel, MeasureOfDispersion, django_setup, get_gov_level_ids


from base.models import (
    AppendInfoToFeaturesPairwiseFlat,
    AppendMacroDataInfoToFeaturesPairwiseFlat,
    CalcValueOntologies,
    ConcatDataFrames,
    CorrelationCalculation,
    DataFrameFile,
    DispersionCalculation,
    DownloadOneGovAcrossTime,
    DownloadOneGovMacroDataAcrossTime,
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
# Time series similarities macro data
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
            in1 = DownloadOneGovMacroDataAcrossTime.objects.create_download_one_gov_macro_data_across_time(
                gov_id=gov_id_once,
                months=months,
            )

            # Calculation
            drop_cols = LowQualityRowsAndColsRemoval.objects.create(
                input=in1.output,
                min_quality_for_col=0.9,
                min_quality_for_row=0,
            )
            corr = CorrelationCalculation.objects.create(
                input=drop_cols.output,
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
            dispersion = DispersionCalculation.objects.create(
                input=drop_cols.output,
                measure_of_dispersion=MeasureOfDispersion.QCD,
            )
            result = AppendMacroDataInfoToFeaturesPairwiseFlat.objects.create(
                input=filtered_corr.output,
                input_dispersion=dispersion.output,
            )

            # Remove cached files
            os.remove(in1.output.cached_file.path)
            os.remove(drop_cols.output.cached_file.path)
            os.remove(corr.output.cached_file.path)
            os.remove(sorted_corr.output.cached_file.path)
            os.remove(filtered_corr.output.cached_file.path)
            os.remove(dispersion.output.cached_file.path)
            os.rename(
                result.output.cached_file.path,
                f"./out/macro_data_time_series/{gov_id_once}.csv",
            )

            return result

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
