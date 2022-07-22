import enum
from typing import Any, Optional, Sequence
from django.core.exceptions import FieldError
import pandas as pd

from base.models import (
    AppendInfoToAssociationRules,
    AppendInfoToFeaturesPairwiseFlat,
    AssociationRules,
    ByIndustryFiltering,
    CalcValueOntologies,
    CorrelationCalculation,
    DataFrameFile,
    DiscretizeAndToOneHot,
    DispersionCalculation,
    DistrictsIncomeLevelFiltering,
    DistrictsPopulationLevelFiltering,
    DistrictsSevenAreasLevelFiltering,
    DownloadOneGovAcrossTime,
    DownloadOneMonth,
    EachColOutliersIdentification,
    EachColPerCapita,
    FillNA,
    FilterByInfo,
    FilterValuesInFeaturesPairwiseFlat,
    FrequentItemSets,
    GovLevelFiltering,
    Job,
    LowQualityRowsAndColsRemoval,
    Month,
    ProvincialCapitalsChildrenFiltering,
    RemoveDispersionSmallCol,
    RemoveTooManyDuplicatesCol,
    SortValuesInFeaturesPairwise,
)
from utils import (
    GovLevel,
    MeasureOfDispersion,
)


class SampleSelectionTechnique(enum.Enum):
    DISTRICTS = 0
    DISTRICTS_POPULATION = 1
    DISTRICTS_INCOME = 2
    DISTRICTS_SEVEN_AREAS = 3
    DISTRICTS_PROVINCIAL_CAPITALS_CHILDREN = 4
    DISTRICTS_BY_INDUSTRY = 5
    TIME_SERIES = 6


class MiningTechnique(enum.Enum):
    PEARSON_CORR = 0
    ASSOCIATION_RULES = 1
    FREQUENT_ITEM_SETS = 2
    TIME_SERIES = 3


SAMPLE_SELECTION_TECHNIQUES: dict[SampleSelectionTechnique, list[tuple]] = {
    SampleSelectionTechnique.DISTRICTS: [
        (
            DownloadOneMonth,
            {
                "month": "month_",
            },
        ),
        (
            GovLevelFiltering,
            {
                "gov_level": GovLevel.DISTRICT_LEVEL.value,
            },
        ),
    ],
    SampleSelectionTechnique.DISTRICTS_POPULATION: [
        (
            DownloadOneMonth,
            {
                "month": "month_",
            },
        ),
        (
            GovLevelFiltering,
            {
                "gov_level": GovLevel.DISTRICT_LEVEL.value,
            },
        ),
        (
            DistrictsPopulationLevelFiltering,
            {
                "pop_level": "pop_level_",
            },
        ),
    ],
    SampleSelectionTechnique.DISTRICTS_INCOME: [
        (
            DownloadOneMonth,
            {
                "month": "month_",
            },
        ),
        (
            GovLevelFiltering,
            {
                "gov_level": GovLevel.DISTRICT_LEVEL.value,
            },
        ),
        (
            DistrictsIncomeLevelFiltering,
            {
                "income_level": "income_level_",
            },
        ),
    ],
    SampleSelectionTechnique.DISTRICTS_SEVEN_AREAS: [
        (
            DownloadOneMonth,
            {
                "month": "month_",
            },
        ),
        (
            GovLevelFiltering,
            {
                "gov_level": GovLevel.DISTRICT_LEVEL.value,
            },
        ),
        (
            DistrictsSevenAreasLevelFiltering,
            {
                "seven_areas_level": "seven_areas_level_",
            },
        ),
    ],
    SampleSelectionTechnique.DISTRICTS_PROVINCIAL_CAPITALS_CHILDREN: [
        (
            DownloadOneMonth,
            {
                "month": "month_",
            },
        ),
        (
            GovLevelFiltering,
            {
                "gov_level": GovLevel.DISTRICT_LEVEL.value,
            },
        ),
        (
            ProvincialCapitalsChildrenFiltering,
            {
                "provincial_capitals_children": True,
            },
        ),
    ],
    SampleSelectionTechnique.DISTRICTS_BY_INDUSTRY: [
        (
            DownloadOneMonth,
            {
                "month": "month_",
            },
        ),
        (
            GovLevelFiltering,
            {
                "gov_level": GovLevel.DISTRICT_LEVEL.value,
            },
        ),
        (
            ByIndustryFiltering,
            {
                "industry": "industry_",
            },
        ),
    ],
    SampleSelectionTechnique.TIME_SERIES: [
        (
            DownloadOneGovAcrossTime,
            {
                "gov_id": "gov_id_",
                "months": "months_",
            },
        ),
    ],
}


MINING_TECHNIQUES: dict[MiningTechnique, Any] = {
    MiningTechnique.PEARSON_CORR: [
        # Calculation
        (
            EachColPerCapita,
            {
                "input": "samples_dff_",
            },
        ),
        (
            EachColOutliersIdentification,
            {
                "deviation_threshold": 5,
            },
        ),
        (
            LowQualityRowsAndColsRemoval,
            {
                "min_quality_for_col": 0.75,
                "min_quality_for_row": 0,
            },
        ),
        (
            LowQualityRowsAndColsRemoval,
            {
                "min_quality_for_col": 0,
                "min_quality_for_row": 0.9,
            },
            "drop_rows",
        ),
        (
            CorrelationCalculation,
            {
                "method": CorrelationCalculation.PEARSON,
            },
        ),
        (
            SortValuesInFeaturesPairwise,
            {},
        ),
        (
            FilterValuesInFeaturesPairwiseFlat,
            {
                "func_before_filter": FilterValuesInFeaturesPairwiseFlat.ABS,
                "min_value": 0.5,
            },
            "filtered_corr",
        ),
        # Append info
        (
            CalcValueOntologies,
            {},
            "value_ontologies",
        ),
        (
            DispersionCalculation,
            {
                "input": "drop_rows.output",
                "measure_of_dispersion": MeasureOfDispersion.CV2,
            },
            "dispersion",
        ),
        (
            AppendInfoToFeaturesPairwiseFlat,
            {
                "input": "filtered_corr.output",
                "input_value_ontologies": "value_ontologies.output",
                "input_dispersion": "dispersion.output",
            },
        ),
        (
            FilterByInfo,
            {
                "no_same_type": True,
                "no_year_period": False,
                "min_dispersion": 0.5,
            },
        ),
    ],
    MiningTechnique.ASSOCIATION_RULES: [
        (
            EachColPerCapita,
            {
                "input": "samples_dff_",
            },
        ),
        (
            EachColOutliersIdentification,
            {
                "deviation_threshold": 5,
            },
        ),
        (
            LowQualityRowsAndColsRemoval,
            {
                "min_quality_for_col": 0.75,
                "min_quality_for_row": 0,
            },
        ),
        (
            LowQualityRowsAndColsRemoval,
            {
                "min_quality_for_col": 0,
                "min_quality_for_row": 0.9,
            },
            "drop_rows",
        ),
        (
            RemoveTooManyDuplicatesCol,
            {
                "max_duplicates_percentage": 0.1,
            },
        ),
        (
            RemoveDispersionSmallCol,
            {
                "measure_of_dispersion": MeasureOfDispersion.CV2,
                "min_dispersion": 10,
            },
        ),
        (
            FillNA,
            {},
        ),
        (
            DiscretizeAndToOneHot,
            {
                "number_of_bins": 3,
            },
        ),
        (
            AssociationRules,
            {
                "max_len": None,
                "min_support": 0.2,
            },
            "association_rules",
        ),
        (
            CalcValueOntologies,
            {},
            "value_ontologies",
        ),
        (
            DispersionCalculation,
            {
                "input": "drop_rows.output",
                "measure_of_dispersion": MeasureOfDispersion.CV2,
            },
            "dispersion",
        ),
        (
            AppendInfoToAssociationRules,
            {
                "input": "association_rules.output",
                "input_value_ontologies": "value_ontologies.output",
                "input_dispersion": "dispersion.output",
                "number_of_bins": 3,
            },
        ),
    ],
    MiningTechnique.FREQUENT_ITEM_SETS: [
        (
            EachColPerCapita,
            {
                "input": "samples_dff_",
            },
        ),
        (
            EachColOutliersIdentification,
            {
                "deviation_threshold": 5,
            },
        ),
        (
            LowQualityRowsAndColsRemoval,
            {
                "min_quality_for_col": 0.75,
                "min_quality_for_row": 0,
            },
        ),
        (
            LowQualityRowsAndColsRemoval,
            {
                "min_quality_for_col": 0,
                "min_quality_for_row": 0.9,
            },
            "drop_rows",
        ),
        (
            RemoveTooManyDuplicatesCol,
            {
                "max_duplicates_percentage": 0.1,
            },
        ),
        (
            RemoveDispersionSmallCol,
            {
                "measure_of_dispersion": MeasureOfDispersion.CV2,
                "min_dispersion": 10,
            },
        ),
        (
            FillNA,
            {},
        ),
        (
            DiscretizeAndToOneHot,
            {
                "number_of_bins": 3,
            },
        ),
        (
            FrequentItemSets,
            {
                "max_len": None,
                "min_support": 0.2,
            },
            "association_rules",
        ),
        (
            CalcValueOntologies,
            {},
            "value_ontologies",
        ),
        (
            DispersionCalculation,
            {
                "input": "drop_rows.output",
                "measure_of_dispersion": MeasureOfDispersion.CV2,
            },
            "dispersion",
        ),
        (
            AppendInfoToAssociationRules,
            {
                "input": "association_rules.output",
                "input_value_ontologies": "value_ontologies.output",
                "input_dispersion": "dispersion.output",
                "number_of_bins": 3,
            },
        ),
    ],
    MiningTechnique.TIME_SERIES: [
        (
            LowQualityRowsAndColsRemoval,
            {
                "input": "samples_dff_",
                "min_quality_for_col": 0.8,
                "min_quality_for_row": 0,
            },
        ),
        (
            RemoveDispersionSmallCol,
            {
                "measure_of_dispersion": MeasureOfDispersion.CV2,
                "min_dispersion": 0.5,
            },
            "remove_dispersion_small_col",
        ),
        (
            CorrelationCalculation,
            {
                "method": CorrelationCalculation.PEARSON,
            },
        ),
        (
            SortValuesInFeaturesPairwise,
            {},
        ),
        (
            FilterValuesInFeaturesPairwiseFlat,
            {
                "func_before_filter": FilterValuesInFeaturesPairwiseFlat.ABS,
                "min_value": 0.5,
            },
            "filtered_corr",
        ),
        (
            CalcValueOntologies,
            {},
            "value_ontologies",
        ),
        (
            DispersionCalculation,
            {
                "input": "remove_dispersion_small_col.output",
                "measure_of_dispersion": MeasureOfDispersion.CV2,
            },
            "dispersion",
        ),
        (
            AppendInfoToFeaturesPairwiseFlat,
            {
                "input": "filtered_corr.output",
                "input_value_ontologies": "value_ontologies.output",
                "input_dispersion": "dispersion.output",
            },
        ),
        (
            FilterByInfo,
            {
                "no_same_type": True,
                "no_year_period": False,
                "min_dispersion": 0.5,
            },
        ),
    ],
}


# Technique execution


def job_filter_wrapper(filtered_first: Any) -> Optional[Job]:
    for job in filtered_first.order_by("-created_at"):
        if job.has_all_outputs_bool():
            return job
    return None


def exec_technique(technique: Sequence[tuple], **kwargs: Any) -> Job:
    last_job = None
    for job_tuple in technique:
        job_model, params, *maybe_name = job_tuple

        # Actual params
        actual_params = {
            key: value if value not in kwargs else kwargs[value]
            for key, value in params.items()
        }

        # One step
        try:
            try:
                if "input" not in actual_params:
                    actual_params["input"] = (
                        last_job.output if last_job is not None else None  # type: ignore
                    )
                step_job = job_filter_wrapper(
                    job_model.objects.filter(**actual_params)
                ) or job_model.objects.create(**actual_params)
            except FieldError:
                del actual_params["input"]
                step_job = job_filter_wrapper(
                    job_model.objects.filter(**actual_params)
                ) or job_model.objects.create(**actual_params)
        except TypeError:
            step_job = job_model.objects.create(**actual_params)

        last_job = step_job
        if maybe_name:
            kwargs[f"{maybe_name[0]}.output"] = step_job.output

    return last_job  # type: ignore
