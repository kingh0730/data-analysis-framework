import streamlit as st
from utils import MeasureOfDispersion, django_setup


from base.models import (
    AppendInfoToAssociationRules,
    AppendInfoToFeaturesPairwiseFlat,
    AssociationRules,
    CalcValueOntologies,
    CorrelationCalculation,
    DataFrameFile,
    DiscretizeAndToOneHot,
    DiscretizeEachCol,
    DispersionCalculation,
    EachColOutliersIdentification,
    EachColPerCapita,
    FillNA,
    FilterValuesInFeaturesPairwiseFlat,
    FrequentItemSets,
    Job,
    LowQualityRowsAndColsRemoval,
    OneMonthGovLevelAndIndexes,
    RemoveDispersionSmallCol,
    RemoveIndexesThatHaveSomeStringInNames,
    RemoveTooManyDuplicatesCol,
    SortValuesInFeaturesPairwise,
)


st.write(
    """
# Association rules
"""
)


with st.sidebar:
    input_df = st.selectbox("Samples and features", DataFrameFile.objects.all())
    number_of_bins = st.radio("Number of bins", range(2, 5), index=1)


with st.echo():

    def main() -> Job:
        # Calculation
        no_da_lei = RemoveIndexesThatHaveSomeStringInNames.objects.create(
            input=input_df,
            some_string="（大类）",
        )
        per_capita = EachColPerCapita.objects.create(
            input=no_da_lei.output,
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
        remove_too_many_duplicates_col = RemoveTooManyDuplicatesCol.objects.create(
            input=drop_rows.output,
            max_duplicates_percentage=0.1,
        )
        remove_dispersion_small_col = RemoveDispersionSmallCol.objects.create(
            input=remove_too_many_duplicates_col.output,
            measure_of_dispersion=MeasureOfDispersion.CV2,
            min_dispersion=0.5,
        )
        fill_na = FillNA.objects.create(
            input=remove_dispersion_small_col.output,
        )
        discretized = DiscretizeAndToOneHot.objects.create(
            input=fill_na.output,
            number_of_bins=number_of_bins,
        )
        association_rules = AssociationRules.objects.create(
            input=discretized.output,
            max_len=None,
            min_support=0.2,
        )
        # association_rules = FrequentItemSets.objects.create(
        #     input=discretized.output,
        #     max_len=None,
        #     min_support=0.2,
        # )

        # Append info
        value_ontologies = CalcValueOntologies.objects.create()
        dispersion = DispersionCalculation.objects.create(
            input=drop_rows.output,
            measure_of_dispersion=MeasureOfDispersion.CV2,
        )
        result = AppendInfoToAssociationRules.objects.create(
            input=association_rules.output,
            input_value_ontologies=value_ontologies.output,
            input_dispersion=dispersion.output,
            number_of_bins=number_of_bins,
        )
        return result


if st.button("Execute main()"):
    st.info("Executing...")
    st.warning("Check terminal.")
    st.success(f"Success: {main()}")
