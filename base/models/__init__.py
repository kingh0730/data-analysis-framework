import enum
from itertools import chain
from typing import Any, Callable, Iterable, Optional
from django.db import models
from django.utils.html import format_html
from django.core.files.storage import default_storage
import numpy as np

from utils import (
    EIGHTY_TWO_INDUSTRIES,
    DistrictsIncomeLevel,
    DistrictsPopulationLevel,
    GovLevel,
    MeasureOfDispersion,
    gov_bar_data,
)
from .managers import (
    DownloadOneGovAcrossTimeManager,
    DownloadOneGovMacroDataAcrossTimeManager,
    JobManager,
    MonthManager,
    DataFrameFileManager,
    TimeSeriesSimilaritiesManager,
)


# Create your models here.


DATA_FRAME_FILE_TYPES = [
    "DataFrameFile",
    "OneMonthGovsAndIndexes",
    "OneMonthGovLevelAndIndexes",
    "FeaturesPairwise",
    "FeaturesPairwiseFlat",
]
"""
There are five data frame file types.

Deprecated:
    You should not add any more data frame file types, because it's unnecessary.
"""


RELATED_NAMES = {
    "MultivariateOutliersRemoval": {
        "input": "input",
        "output": "output",
    },
    "EachColOutliersIdentification": {
        "input": "input_each_col_outliers_identification",
        "output": "output_each_col_outliers_identification",
    },
    "TwoDataFramesComparison": {
        "input1": "input1_two_data_frames_comparison",
        "input2": "input2_two_data_frames_comparison",
        "output": "output_two_data_frames_comparison",
    },
    "GovLevelFiltering": {
        "input": "input_gov_level_filtering",
        "output": "output_gov_level_filtering",
    },
    "LowQualityColsRemoval": {
        "input": "input_low_quality_cols_removal",
        "output": "output_low_quality_cols_removal",
    },
    "CorrelationCalculation": {
        "input": "input_correlation_calculation",
        "output": "output_correlation_calculation",
    },
    "SortValuesInFeaturesPairwise": {
        "input": "input_sort_values_in_features_pairwise",
        "output": "output_sort_values_in_features_pairwise",
    },
    "DispersionCalculation": {
        "input": "input_dispersion_calculation",
        "output": "output_dispersion_calculation",
    },
    "FilterValuesInFeaturesPairwiseFlat": {
        "input": "input_filter_values_in_features_pairwise_flat",
        "output": "output_filter_values_in_features_pairwise_flat",
    },
    "CountOccurrencesInFeaturesPairwiseFlat": {
        "input": "input_count_occurrences_in_features_pairwise_flat",
        "output": "output_count_occurrences_in_features_pairwise_flat",
    },
    "PCA": {
        "input": "input_pca",
        "output": "output_pca",
    },
    "EachColPerCapita": {
        "input": "input_each_col_per_capita",
        "output": "output_each_col_per_capita",
    },
    "AppendInfoToFeaturesPairwiseFlat": {
        "input": "input_append_info_to_features_pairwise_flat",
        "output": "output_append_info_to_features_pairwise_flat",
        "input_value_ontologies": "input_value_ontologies_append_info_to_features_pairwise_flat",
        "input_dispersion": "input_dispersion_append_info_to_features_pairwise_flat",
    },
    "DownloadOneMonth": {
        "output": "output_download_one_month",
    },
    "TimeSeriesSimilarities": {
        "input_months": "input_months_time_series_similarities",
        "output": "output_time_series_similarities",
    },
    "EachColRelativeOrdering": {
        "input": "input_each_col_relative_ordering",
        "output": "output_each_col_relative_ordering",
    },
    "DistanceBetweenColPairs": {
        "input": "input_distance_between_col_pairs",
        "output": "output_distance_between_col_pairs",
    },
    "LowQualityRowsAndColsRemoval": {
        "input": "input_low_quality_rows_and_cols_removal",
        "output": "output_low_quality_rows_and_cols_removal",
    },
    "FillNA": {
        "input": "input_fill_na",
        "output": "output_fill_na",
    },
    "CalcValueOntologies": {
        "output": "output_calc_value_ontologies",
    },
    "DownloadOneGovAcrossTime": {
        "months": "months_download_one_gov_across_time",
        "output": "output_download_one_gov_across_time",
    },
    "FilterByInfo": {
        "input": "input_filter_by_info",
        "output": "output_filter_by_info",
    },
    "ConcatDataFrames": {
        "input1": "input1_concat_data_frames",
        "input2": "input2_concat_data_frames",
        "output": "output_concat_data_frames",
    },
    "DownloadOneGovMacroDataAcrossTime": {
        "months": "months_download_one_gov_macro_data_across_time",
        "output": "output_download_one_gov_macro_data_across_time",
    },
    "AppendMacroDataInfoToFeaturesPairwiseFlat": {
        "input": "input_append_macro_data_info_to_features_pairwise_flat",
        "output": "output_append_macro_data_info_to_features_pairwise_flat",
        "input_dispersion": "input_dispersion_append_macro_data_info_to_features_pairwise_flat",
    },
    "DiscretizeEachCol": {
        "input": "input_discretize_each_col",
        "output": "output_discretize_each_col",
    },
    "AssociationRules": {
        "input": "input_association_rules",
        "output": "output_association_rules",
    },
    "DynamicTimeWarping": {
        "input": "input_dynamic_time_warping",
        "output": "output_dynamic_time_warping",
    },
    "NormalizeEachCol": {
        "input": "input_normalize_each_col",
        "output": "output_normalize_each_col",
    },
    "DiscretizeAndToOneHot": {
        "input": "input_discretize_and_to_one_hot",
        "output": "output_discretize_and_to_one_hot",
    },
    "RemoveDispersionSmallCol": {
        "input": "input_remove_dispersion_small_col",
        "output": "output_remove_dispersion_small_col",
    },
    "RemoveTooManyDuplicatesCol": {
        "input": "input_remove_too_many_duplicates_col",
        "output": "output_remove_too_many_duplicates_col",
    },
    "AppendInfoToAssociationRules": {
        "input": "input_append_into_association_rules",
        "output": "output_append_into_association_rules",
        "input_value_ontologies": "input_value_ontologies_append_info_to_association_rules",
        "input_dispersion": "input_dispersion_append_info_to_association_rules",
    },
    "FrequentItemSets": {
        "input": "input_frequent_item_sets",
        "output": "output_frequent_item_sets",
    },
    "RemoveIndexesThatHaveSomeStringInNames": {
        "input": "input_remove_indexes_that_have_some_string_in_names",
        "output": "output_remove_indexes_that_have_some_string_in_names",
    },
    "KMeans": {
        "input": "input_k_means",
        "output": "output_k_means",
    },
    "FilterByValueInOneCol": {
        "input": "input_filter_by_value_in_one_col",
        "output": "output_filter_by_value_in_one_col",
    },
    "DistrictsPopulationLevelFiltering": {
        "input": "input_districts_population_level_filtering",
        "output": "output_districts_population_level_filtering",
    },
    "DistrictsIncomeLevelFiltering": {
        "input": "input_districts_income_level_filtering",
        "output": "output_districts_income_level_filtering",
    },
    "DistrictsSevenAreasLevelFiltering": {
        "input": "input_districts_seven_areas_level_filtering",
        "output": "output_districts_seven_areas_level_filtering",
    },
    "ProvincialCapitalsChildrenFiltering": {
        "input": "input_providential_capitals_children_filtering",
        "output": "output_providential_capitals_children_filtering",
    },
    "ByIndustryFiltering": {
        "input": "input_by_industry_filtering",
        "output": "output_by_industry_filtering",
    },
}


def _all_related_names() -> tuple[list[str], list[str]]:
    all_inputs_and_outputs_separate = [
        inputs_and_outputs.values() for inputs_and_outputs in RELATED_NAMES.values()
    ]
    all_inputs_and_outputs = list(chain.from_iterable(all_inputs_and_outputs_separate))
    inputs = [put for put in all_inputs_and_outputs if put[:5] == "input"]
    outputs = [put for put in all_inputs_and_outputs if put[:6] == "output"]

    return inputs, outputs


ALL_INPUTS_RELATED_NAMES, ALL_OUTPUTS_RELATED_NAMES = _all_related_names()


# Choices


GOV_LEVEL_CHOICES = [(gov_level.value, gov_level.name) for gov_level in GovLevel]
DISTRICTS_POPULATION_LEVEL_CHOICES = [
    (pop_level.value, pop_level.name) for pop_level in DistrictsPopulationLevel
]
DISTRICTS_INCOME_LEVEL_CHOICES = [
    (income_level.value, income_level.name) for income_level in DistrictsIncomeLevel
]
SEVEN_AREAS_CHOICES = list(
    enumerate([area_dict["name"] for area_dict in gov_bar_data.area_conf])
)
EIGHTY_TWO_INDUSTRY_CHOICES = [
    (row["daas_id"], row["type_name"]) for _, row in EIGHTY_TWO_INDUSTRIES.iterrows()
]


METRIC_KINDS = [("euclidean", "euclidean"), ("cityblock", "cityblock")]


MEASURES_OF_DISPERSION = [(item.name, item.value) for item in MeasureOfDispersion]


class Month(models.Model):
    month_int = models.PositiveSmallIntegerField(primary_key=True)
    month_str = models.CharField(max_length=7, unique=True)

    objects = MonthManager()

    def __str__(self) -> str:
        return f"{self.month_str}({self.month_int})"

    class Meta:
        verbose_name_plural = "0.a. Months"


class DataFrameFile(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)

    cached_file = models.FileField(null=True, blank=True)
    history = models.JSONField(blank=True, default=list)

    objects = DataFrameFileManager()

    def get_data_frame_file_type(self) -> str:
        this_dff_types = [
            dff_type
            for dff_type in DATA_FRAME_FILE_TYPES
            if hasattr(self, dff_type.lower())
        ]
        return this_dff_types[0] if this_dff_types else DATA_FRAME_FILE_TYPES[0]

    def get_specific_data_frame_file(self) -> models.Model:
        dff_type = self.get_data_frame_file_type()
        specific_dff = (
            self
            if dff_type == DATA_FRAME_FILE_TYPES[0]
            else getattr(self, dff_type.lower())
        )
        return specific_dff

    def get_upper_job_type(self) -> Optional[str]:
        specific_dff = self.get_specific_data_frame_file()
        this_output_types = [
            output_type
            for output_type in ALL_OUTPUTS_RELATED_NAMES
            if hasattr(specific_dff, output_type)
        ]
        return this_output_types[0] if this_output_types else None

    def get_upper_job(self) -> Optional[models.Model]:
        specific_dff = self.get_specific_data_frame_file()
        this_output_types = [
            output_type
            for output_type in ALL_OUTPUTS_RELATED_NAMES
            if hasattr(specific_dff, output_type)
        ]
        return (
            getattr(specific_dff, this_output_types[0]) if this_output_types else None
        )

    def get_upper_job_id(self) -> Optional[int]:
        upper_job = self.get_upper_job()
        return upper_job.id if upper_job else None

    def file_exists_bool(self) -> bool:
        result = (
            default_storage.exists(self.cached_file.name) if self.cached_file else False
        )
        return result

    def file_exists(self) -> str:
        result = self.file_exists_bool()
        return format_html(
            """
        <span style="font-weight: bold; color: {1}">{0}</span>
        """,
            result,
            "green" if result else "red",
        )

    def __str__(self) -> str:
        return f"{self.cached_file}({self.created_at})"

    class Meta:
        verbose_name_plural = "1. Data frame files"


class OneMonthGovsAndIndexes(DataFrameFile):
    month = models.ForeignKey(Month, on_delete=models.PROTECT)

    def __str__(self) -> str:
        return f"{str(self.month)}({self.created_at})"

    class Meta:
        verbose_name_plural = "1.a. One month govs and indexes"


class OneMonthGovLevelAndIndexes(DataFrameFile):
    month = models.ForeignKey(Month, on_delete=models.PROTECT)
    gov_level = models.SmallIntegerField(choices=GOV_LEVEL_CHOICES)

    def __str__(self) -> str:
        return f"{str(self.month)}_{GovLevel(self.gov_level)}({self.created_at})"

    class Meta:
        verbose_name_plural = "1.b. One month gov level and indexes"


class FeaturesPairwise(DataFrameFile):
    class Meta:
        verbose_name_plural = "1.c. Features pairwise"


class FeaturesPairwiseFlat(DataFrameFile):
    class Meta:
        verbose_name_plural = "1.d. Features pairwise flat"


class Job(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)

    progress = models.PositiveSmallIntegerField(default=0, blank=True)

    objects = JobManager()

    def get_job_type(self) -> str:
        (this_job_type,) = [
            job_type for job_type in RELATED_NAMES if hasattr(self, job_type.lower())
        ]
        return this_job_type

    def get_specific_job(self) -> models.Model:
        job_type = self.get_job_type()
        specific_job = getattr(self, job_type.lower())
        return specific_job

    def has_all_outputs_bool(self) -> bool:
        job_type = self.get_job_type()
        specific_job = getattr(self, job_type.lower())
        result = all(
            (
                getattr(specific_job, k)
                for k in RELATED_NAMES[job_type]
                if k[:6] == "output"
            )
        )
        return result

    def has_all_outputs(self) -> str:
        result = self.has_all_outputs_bool()
        return format_html(
            """
        <span style="font-weight: bold; color: {1};">{0}</span>
        """,
            result,
            "green" if result else "red",
        )

    def get_all_inputs(
        self,
    ) -> dict[str, DataFrameFile | Iterable[DataFrameFile] | None]:
        job_type = self.get_job_type()
        specific_job = getattr(self, job_type.lower())

        def expand_m2m(
            input_or_inputs: DataFrameFile | Any | None,
        ) -> DataFrameFile | Iterable[DataFrameFile] | None:
            if input_or_inputs is None:
                return None
            if isinstance(input_or_inputs, DataFrameFile):
                return input_or_inputs
            return input_or_inputs.all()

        return {
            k: expand_m2m(getattr(specific_job, k))
            for k in RELATED_NAMES[job_type]
            if k[:5] == "input"
        }

    class Meta:
        verbose_name_plural = "2. Jobs"

    def percentage_progress(self) -> str:
        return format_html(
            """
        <progress value="{0}" max="100"></progress>
        <span>{0}%</span>
        """,
            self.progress,
        )


class MultivariateOutliersRemoval(Job):
    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["MultivariateOutliersRemoval"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["MultivariateOutliersRemoval"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.z. Todo: Multivariate outliers removal"


class EachColOutliersIdentification(Job):

    deviation_threshold = models.FloatField()

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["EachColOutliersIdentification"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["EachColOutliersIdentification"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.a. Each col outliers identification"


class TwoDataFramesComparison(Job):
    input1 = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["TwoDataFramesComparison"]["input1"],
    )
    input2 = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["TwoDataFramesComparison"]["input2"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["TwoDataFramesComparison"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.b. Two data frames comparison"


class GovLevelFiltering(Job):
    gov_level = models.SmallIntegerField(choices=GOV_LEVEL_CHOICES)

    input = models.ForeignKey(
        OneMonthGovsAndIndexes,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["GovLevelFiltering"]["input"],
    )
    output = models.OneToOneField(
        OneMonthGovLevelAndIndexes,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["GovLevelFiltering"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.c. Gov level filtering"


class LowQualityColsRemoval(Job):
    min_count = models.PositiveBigIntegerField(null=True, blank=True)
    min_quality = models.FloatField(null=True, blank=True)

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["LowQualityColsRemoval"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["LowQualityColsRemoval"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.d. Low quality cols removal"

        constraints = [
            models.CheckConstraint(
                name="one and only one - min_count or min_quality",
                check=(
                    models.Q(min_count__isnull=True, min_quality__isnull=False)
                    | models.Q(min_count__isnull=False, min_quality__isnull=True)
                ),
            )
        ]


class CorrelationCalculation(Job):
    PEARSON = "pearson"
    KENDALL = "kendall"
    SPEARMAN = "spearman"
    METHODS_OF_CORRELATION = [
        (PEARSON, PEARSON),
        (KENDALL, KENDALL),
        (SPEARMAN, SPEARMAN),
    ]

    method = models.CharField(
        max_length=15,
        choices=METHODS_OF_CORRELATION,
        default=PEARSON,
    )

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["CorrelationCalculation"]["input"],
    )
    output = models.OneToOneField(
        FeaturesPairwise,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["CorrelationCalculation"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.e. Correlation calculation"


class SortValuesInFeaturesPairwise(Job):
    input = models.ForeignKey(
        FeaturesPairwise,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["SortValuesInFeaturesPairwise"]["input"],
    )
    output = models.OneToOneField(
        FeaturesPairwiseFlat,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["SortValuesInFeaturesPairwise"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.f. Sort values in features pairwise"


class DispersionCalculation(Job):
    measure_of_dispersion = models.CharField(
        max_length=3, choices=MEASURES_OF_DISPERSION
    )

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["DispersionCalculation"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["DispersionCalculation"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.g. Dispersion calculation"


class FilterValuesInFeaturesPairwiseFlat(Job):
    ID = "ID"
    ABS = "ABS"

    FUNC_BEFORE_FILTER = [
        (ID, "Identity"),
        (ABS, "Absolute value"),
    ]
    FUNC_LAMBDAS: dict[str, Callable] = {
        ID: lambda x: x,
        ABS: np.abs,
    }
    func_before_filter = models.CharField(
        max_length=3, choices=FUNC_BEFORE_FILTER, default=ID
    )

    min_value = models.FloatField(null=True, blank=True)
    max_value = models.FloatField(null=True, blank=True)

    input = models.ForeignKey(
        FeaturesPairwiseFlat,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["FilterValuesInFeaturesPairwiseFlat"]["input"],
    )
    output = models.OneToOneField(
        FeaturesPairwiseFlat,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["FilterValuesInFeaturesPairwiseFlat"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.h. Filter values in features pairwise flat"

        constraints = [
            models.CheckConstraint(
                name="",
                check=(~models.Q(min_value__isnull=True, max_value__isnull=True)),
            )
        ]


class CountOccurrencesInFeaturesPairwiseFlat(Job):
    input = models.ForeignKey(
        FeaturesPairwiseFlat,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["CountOccurrencesInFeaturesPairwiseFlat"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["CountOccurrencesInFeaturesPairwiseFlat"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.i. Count occurrences in features pairwise flat"


class PCA(Job):
    index_cols_count = models.PositiveSmallIntegerField(blank=True, default=1)

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["PCA"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["PCA"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.j. PCA"


class EachColPerCapita(Job):
    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["EachColPerCapita"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["EachColPerCapita"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.k. Each col per capita"


class AppendInfoToFeaturesPairwiseFlat(Job):
    has_input_comments = models.BooleanField(default=False)

    input_value_ontologies = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["AppendInfoToFeaturesPairwiseFlat"][
            "input_value_ontologies"
        ],
        blank=True,
        null=True,
    )
    input_dispersion = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["AppendInfoToFeaturesPairwiseFlat"][
            "input_dispersion"
        ],
        blank=True,
        null=True,
    )

    input = models.ForeignKey(
        FeaturesPairwiseFlat,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["AppendInfoToFeaturesPairwiseFlat"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["AppendInfoToFeaturesPairwiseFlat"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.l. Append info to features pairwise flat"


class DownloadOneMonth(Job):
    month = models.ForeignKey(Month, on_delete=models.PROTECT)

    output = models.OneToOneField(
        OneMonthGovsAndIndexes,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["DownloadOneMonth"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.m. Download one month"


class TimeSeriesSimilarities(Job):
    input_months = models.ManyToManyField(
        OneMonthGovsAndIndexes,
        related_name=RELATED_NAMES["TimeSeriesSimilarities"]["input_months"],
    )

    output = models.OneToOneField(
        FeaturesPairwise,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["TimeSeriesSimilarities"]["output"],
        null=True,
        blank=True,
    )

    objects = TimeSeriesSimilaritiesManager()

    class Meta:
        verbose_name_plural = "2.n. Time series similarities"


class EachColRelativeOrdering(Job):
    ORD = "ORD"
    PER = "PER"

    METHODS_OF_RELATIVE_ORDERING = [
        (ORD, "Ordinal"),
        (PER, "Percentage"),
    ]

    method_of_relative_ordering = models.CharField(
        max_length=3,
        choices=METHODS_OF_RELATIVE_ORDERING,
        default=ORD,
    )

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["EachColRelativeOrdering"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["EachColRelativeOrdering"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.o. Each col relative ordering"


class DistanceBetweenColPairs(Job):
    metric = models.CharField(max_length=255, choices=METRIC_KINDS)

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["DistanceBetweenColPairs"]["input"],
    )
    output = models.OneToOneField(
        FeaturesPairwise,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["DistanceBetweenColPairs"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.p. Distance between col pairs"


class LowQualityRowsAndColsRemoval(Job):
    min_quality_for_row = models.FloatField()
    min_quality_for_col = models.FloatField()

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["LowQualityRowsAndColsRemoval"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["LowQualityRowsAndColsRemoval"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.q. Low quality rows and cols removal"


class FillNA(Job):
    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["FillNA"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["FillNA"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.r. Fill NA"


class CalcValueOntologies(Job):
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["CalcValueOntologies"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.s. Calc value ontologies"


class DownloadOneGovAcrossTime(Job):
    gov_id = models.PositiveSmallIntegerField()
    months = models.ManyToManyField(
        Month, related_name=RELATED_NAMES["DownloadOneGovAcrossTime"]["months"]
    )

    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["DownloadOneGovAcrossTime"]["output"],
        null=True,
        blank=True,
    )

    objects = DownloadOneGovAcrossTimeManager()

    class Meta:
        verbose_name_plural = "2.t. Download one gov across time"


class FilterByInfo(Job):
    no_same_type = models.BooleanField(default=False)
    no_year_period = models.BooleanField(default=False)
    min_dispersion = models.FloatField(null=True, blank=True)

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["FilterByInfo"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["FilterByInfo"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.u. Filter by info"


class ConcatDataFrames(Job):
    comment = models.CharField(max_length=255)
    """
    Should not include "---"
    """

    input1 = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["ConcatDataFrames"]["input1"],
    )
    input2 = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["ConcatDataFrames"]["input2"],
    )

    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["ConcatDataFrames"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.v. Concat data frames"


class DownloadOneGovMacroDataAcrossTime(Job):
    gov_id = models.PositiveSmallIntegerField()
    months = models.ManyToManyField(
        Month, related_name=RELATED_NAMES["DownloadOneGovMacroDataAcrossTime"]["months"]
    )

    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["DownloadOneGovMacroDataAcrossTime"]["output"],
        null=True,
        blank=True,
    )

    objects = DownloadOneGovMacroDataAcrossTimeManager()

    class Meta:
        verbose_name_plural = "2.w. Download one gov macro data across time"


class AppendMacroDataInfoToFeaturesPairwiseFlat(Job):
    has_input_comments = models.BooleanField(default=False)

    input_dispersion = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["AppendMacroDataInfoToFeaturesPairwiseFlat"][
            "input_dispersion"
        ],
        blank=True,
        null=True,
    )

    input = models.ForeignKey(
        FeaturesPairwiseFlat,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["AppendMacroDataInfoToFeaturesPairwiseFlat"][
            "input"
        ],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["AppendMacroDataInfoToFeaturesPairwiseFlat"][
            "output"
        ],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.x. Append macro data info to features pairwise flat"


class DiscretizeEachCol(Job):
    number_of_bins = models.PositiveSmallIntegerField(blank=True, default=3)

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["DiscretizeEachCol"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["DiscretizeEachCol"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.y. Discretize each col"


class AssociationRules(Job):
    max_len = models.PositiveSmallIntegerField(null=True, blank=True)
    min_support = models.FloatField()

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["AssociationRules"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["AssociationRules"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.A. Association rules"


class DynamicTimeWarping(Job):
    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["DynamicTimeWarping"]["input"],
    )
    output = models.OneToOneField(
        FeaturesPairwise,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["DynamicTimeWarping"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.B. Dynamic time warping"


class NormalizeEachCol(Job):
    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["NormalizeEachCol"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["NormalizeEachCol"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.C. Normalize each col"


class DiscretizeAndToOneHot(Job):
    number_of_bins = models.PositiveSmallIntegerField(blank=True, default=3)

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["DiscretizeAndToOneHot"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["DiscretizeAndToOneHot"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.D. Discretize and to one hot"


class RemoveDispersionSmallCol(Job):
    measure_of_dispersion = models.CharField(
        max_length=3, choices=MEASURES_OF_DISPERSION
    )
    min_dispersion = models.FloatField()

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["RemoveDispersionSmallCol"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["RemoveDispersionSmallCol"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.E. Remove dispersion small col"


class RemoveTooManyDuplicatesCol(Job):
    max_duplicates_percentage = models.FloatField()

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["RemoveTooManyDuplicatesCol"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["RemoveTooManyDuplicatesCol"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.F. Remove too many duplicates col"


class AppendInfoToAssociationRules(Job):
    number_of_bins = models.PositiveSmallIntegerField()

    input_value_ontologies = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["AppendInfoToAssociationRules"][
            "input_value_ontologies"
        ],
        blank=True,
        null=True,
    )
    input_dispersion = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["AppendInfoToAssociationRules"]["input_dispersion"],
        blank=True,
        null=True,
    )

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["AppendInfoToAssociationRules"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["AppendInfoToAssociationRules"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.Z. Append info to association rules"


class FrequentItemSets(Job):
    max_len = models.PositiveSmallIntegerField(null=True, blank=True)
    min_support = models.FloatField()

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["FrequentItemSets"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["FrequentItemSets"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.G. Frequent item sets"


class RemoveIndexesThatHaveSomeStringInNames(Job):
    some_string = models.TextField()

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["RemoveIndexesThatHaveSomeStringInNames"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["RemoveIndexesThatHaveSomeStringInNames"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.H. Remove indexes that have some string in names"


class KMeans(Job):
    n_clusters = models.PositiveSmallIntegerField()
    use_cols = models.TextField(null=False, blank=True)

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["KMeans"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["KMeans"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.I. K-means"


class FilterByValueInOneCol(Job):
    use_col = models.CharField(max_length=255)
    min_value = models.FloatField(null=True, blank=True)
    max_value = models.FloatField(null=True, blank=True)

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["FilterByValueInOneCol"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["FilterByValueInOneCol"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.J. Filter by value in one col"


class DistrictsPopulationLevelFiltering(Job):
    pop_level = models.SmallIntegerField(choices=DISTRICTS_POPULATION_LEVEL_CHOICES)

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["DistrictsPopulationLevelFiltering"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["DistrictsPopulationLevelFiltering"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.K. Districts population level filtering"


class DistrictsIncomeLevelFiltering(Job):
    income_level = models.SmallIntegerField(choices=DISTRICTS_INCOME_LEVEL_CHOICES)

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["DistrictsIncomeLevelFiltering"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["DistrictsIncomeLevelFiltering"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.L. Districts income level filtering"


class DistrictsSevenAreasLevelFiltering(Job):
    seven_areas_level = models.SmallIntegerField(choices=SEVEN_AREAS_CHOICES)

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["DistrictsSevenAreasLevelFiltering"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["DistrictsSevenAreasLevelFiltering"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.M. Districts seven areas level filtering"


class ProvincialCapitalsChildrenFiltering(Job):
    provincial_capitals_children = models.BooleanField()

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["ProvincialCapitalsChildrenFiltering"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["ProvincialCapitalsChildrenFiltering"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.N. Provincial capitals children filtering"


class ByIndustryFiltering(Job):
    industry = models.PositiveIntegerField(choices=EIGHTY_TWO_INDUSTRY_CHOICES)

    input = models.ForeignKey(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["ByIndustryFiltering"]["input"],
    )
    output = models.OneToOneField(
        DataFrameFile,
        on_delete=models.PROTECT,
        related_name=RELATED_NAMES["ByIndustryFiltering"]["output"],
        null=True,
        blank=True,
    )

    class Meta:
        verbose_name_plural = "2.O. By industry filtering"
