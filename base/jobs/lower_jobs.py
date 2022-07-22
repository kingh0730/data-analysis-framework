from typing import Any, Callable
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn import decomposition
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth


from django.db import transaction
from django.db.models.signals import post_save, m2m_changed
from django.dispatch import receiver


from utils import (
    DAAS_INDEX_LIST,
    MACRO_DATA_DIR_INDEX_NODES,
    SECURE_KEY,
    VALUE_ONTOLOGY_CAN_BE_DIVIDED_BY_POPULATION,
    GovLevel,
    ValueOntology,
    calc_dispersions,
    calc_relative_positions,
    calc_value_ontology,
    get_gov_level_ids,
    get_values_one_month,
    month_int_to_str,
    remove_columns_that_have_too_many_nas,
    remove_outliers_series_with_median_deviation,
    sort_corr,
    sort_series_and_calc_ordinal_numbers,
    zk2861api,
)
from utils.macro_data_api import (
    api_get_macro_data_gov_nodes,
    sync_api_get_macro_data_gov_nodes,
)


from base.models import (
    PCA,
    AppendInfoToAssociationRules,
    AppendInfoToFeaturesPairwiseFlat,
    AppendMacroDataInfoToFeaturesPairwiseFlat,
    AssociationRules,
    CalcValueOntologies,
    ConcatDataFrames,
    CorrelationCalculation,
    CountOccurrencesInFeaturesPairwiseFlat,
    DataFrameFile,
    DiscretizeEachCol,
    DispersionCalculation,
    DistanceBetweenColPairs,
    DownloadOneGovAcrossTime,
    DownloadOneGovMacroDataAcrossTime,
    DownloadOneMonth,
    DynamicTimeWarping,
    EachColPerCapita,
    EachColRelativeOrdering,
    FeaturesPairwise,
    FeaturesPairwiseFlat,
    FillNA,
    FilterByInfo,
    FilterValuesInFeaturesPairwiseFlat,
    GovLevelFiltering,
    LowQualityColsRemoval,
    LowQualityRowsAndColsRemoval,
    Month,
    MultivariateOutliersRemoval,
    EachColOutliersIdentification,
    OneMonthGovLevelAndIndexes,
    OneMonthGovsAndIndexes,
    SortValuesInFeaturesPairwise,
    TimeSeriesSimilarities,
    TwoDataFramesComparison,
)


def on_transaction_commit(func: Callable) -> Callable:
    def inner(*args: Any, **kwargs: Any) -> None:
        transaction.on_commit(lambda: func(*args, **kwargs))

    return inner


# Signals


@receiver(post_save, sender=MultivariateOutliersRemoval)
def create_multivariate_outliers_removal_job(
    sender: Any, instance: MultivariateOutliersRemoval, created: bool, **kwargs: Any
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        instance.progress = 100
        instance.save()
        print("Todo: created!!")


@receiver(post_save, sender=EachColOutliersIdentification)
def create_each_col_outliers_identification_job(
    sender: Any, instance: EachColOutliersIdentification, created: bool, **kwargs: Any
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False

        input_df = (
            instance.in_memory_inputs["input"]
            if "input" in instance.in_memory_inputs
            else pd.read_csv(instance.input.cached_file.path, index_col=0)
        )

        deviation_threshold = instance.deviation_threshold

        # Remove outliers in each column
        output_df = input_df.apply(
            lambda s: remove_outliers_series_with_median_deviation(
                s, deviation_threshold
            )
        )

        # for i, col in enumerate(input_df.columns):
        #     print(f"Progress: {i + 1} / {len(input_df.columns)}")
        #     input_df[col] = remove_outliers_series_with_median_deviation(
        #         input_df[col], instance.deviation_threshold
        #     )

        if instance.should_output_in_memory:
            instance.in_memory_outputs["output"] = output_df

        instance.output = (
            DataFrameFile.objects.create()
            if not instance.should_output_in_file
            else DataFrameFile.objects.create_data_frame_file(
                data_frame=output_df,
                # predecessor_data_frame_file=input_instance,
                job=instance,
            )
        )

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=TwoDataFramesComparison)
def create_two_data_frames_comparison_job(
    sender: Any, instance: TwoDataFramesComparison, created: bool, **kwargs: Any
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        input1: DataFrameFile = instance.input1
        input2: DataFrameFile = instance.input2
        input1_df = pd.read_csv(input1.cached_file.path, index_col=0)
        input2_df = pd.read_csv(input2.cached_file.path, index_col=0)

        # Compare to True and False
        output_df = (input1_df == input2_df) | (
            (input1_df != input1_df) & (input2_df != input2_df)
        )

        output_instance = DataFrameFile.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=GovLevelFiltering)
def create_gov_level_filtering_job(
    sender: Any, instance: GovLevelFiltering, created: bool, **kwargs: Any
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        input_instance: OneMonthGovsAndIndexes = instance.input
        input_df = pd.read_csv(input_instance.cached_file.path, index_col=0)

        month = input_instance.month
        gov_level = instance.gov_level

        # Filtering
        filtered = input_df[input_df["gov_id"].isin(get_gov_level_ids(gov_level))]
        filtered.reset_index(inplace=True, drop=True)

        output_instance = OneMonthGovLevelAndIndexes.objects.create_data_frame_file(
            data_frame=filtered, job=instance, month=month, gov_level=gov_level
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=LowQualityColsRemoval)
def create_low_quality_cols_removal_job(
    sender: Any, instance: LowQualityColsRemoval, created: bool, **kwargs: Any
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False

        input_df = (
            instance.in_memory_inputs["input"]
            if "input" in instance.in_memory_inputs
            else pd.read_csv(instance.input.cached_file.path, index_col=0)
        )

        min_count = instance.min_count
        min_quality = instance.min_quality

        real_mint_count = min_count if min_count else min_quality * len(input_df)

        # Removal
        output_df = remove_columns_that_have_too_many_nas(input_df, real_mint_count)

        if instance.should_output_in_memory:
            instance.in_memory_outputs["output"] = output_df

        instance.output = (
            DataFrameFile.objects.create()
            if not instance.should_output_in_file
            else DataFrameFile.objects.create_data_frame_file(
                data_frame=output_df,
                # predecessor_data_frame_file=input_instance,
                job=instance,
            )
        )

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=CorrelationCalculation)
def create_correlation_calculation_job(
    sender: Any, instance: CorrelationCalculation, created: bool, **kwargs: Any
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        input_instance: DataFrameFile = instance.input
        input_df = pd.read_csv(input_instance.cached_file.path, index_col=0)

        method = instance.method

        # Correlation calculation
        output_df = input_df.corr(method=method)

        output_instance = FeaturesPairwise.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=SortValuesInFeaturesPairwise)
def create_sort_values_in_features_pairwise_job(
    sender: Any, instance: SortValuesInFeaturesPairwise, created: bool, **kwargs: Any
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        input_instance: FeaturesPairwise = instance.input
        input_df = pd.read_csv(input_instance.cached_file.path, index_col=0)

        # Sort
        output_df = sort_corr(input_df)

        output_instance = FeaturesPairwiseFlat.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=DispersionCalculation)
def create_dispersion_calculation_job(
    sender: Any, instance: DispersionCalculation, created: bool, **kwargs: Any
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        input_instance: DataFrameFile = instance.input
        input_df = pd.read_csv(input_instance.cached_file.path, index_col=0)

        measure_of_dispersion = instance.measure_of_dispersion

        # Measure of dispersion
        output_df = calc_dispersions(input_df, measure_of_dispersion)

        output_instance = DataFrameFile.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=FilterValuesInFeaturesPairwiseFlat)
def create_filter_values_in_features_pairwise_flat_job(
    sender: Any,
    instance: FilterValuesInFeaturesPairwiseFlat,
    created: bool,
    **kwargs: Any,
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        input_instance: FeaturesPairwiseFlat = instance.input
        input_df = pd.read_csv(input_instance.cached_file.path, index_col=[0, 1])

        func_before_filter = instance.func_before_filter
        func_lambda = FilterValuesInFeaturesPairwiseFlat.FUNC_LAMBDAS[
            func_before_filter
        ]

        min_value = instance.min_value if instance.min_value else -np.inf
        max_value = instance.max_value if instance.max_value else np.inf

        # Filtering
        output_df = (
            input_df.where(lambda x: func_lambda(x) > min_value)
            .where(lambda x: func_lambda(x) < max_value)
            .dropna()
        )

        output_instance = FeaturesPairwiseFlat.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=CountOccurrencesInFeaturesPairwiseFlat)
def create_count_occurrences_in_features_pairwise_flat_job(
    sender: Any,
    instance: CountOccurrencesInFeaturesPairwiseFlat,
    created: bool,
    **kwargs: Any,
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        input_instance: FeaturesPairwiseFlat = instance.input
        input_df = pd.read_csv(
            input_instance.cached_file.path, index_col=[0, 1], squeeze=True
        )

        # Count
        level_0 = input_df.groupby(level=0).count()
        level_1 = input_df.groupby(level=1).count()
        output_df = level_0.add(level_1, fill_value=0).sort_values(ascending=False)

        output_instance = DataFrameFile.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=PCA)
def create_pca_job(sender: Any, instance: PCA, created: bool, **kwargs: Any) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        input_instance: DataFrameFile = instance.input
        index_cols_count = instance.index_cols_count
        input_df = pd.read_csv(
            input_instance.cached_file.path, index_col=list(range(index_cols_count))
        )

        # PCA
        scaled = (input_df - input_df.mean()) / input_df.std()
        dropped_nas = scaled.dropna(axis=1).dropna(axis=0)
        pca = decomposition.PCA()
        x_pca = pca.fit_transform(dropped_nas)
        output_df = pd.DataFrame(x_pca, index=dropped_nas.index)

        output_instance = DataFrameFile.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=EachColPerCapita)
def create_each_col_per_capita_job(
    sender: Any, instance: EachColPerCapita, created: bool, **kwargs: Any
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        input_instance: DataFrameFile = instance.input
        input_df = pd.read_csv(
            input_instance.cached_file.path, index_col=list(range(3))
        )

        # Constants

        # 常住人口（全国第七次人口普查）
        # POP_7TH_INDEX = 71615

        # 常住人口（年度）（运营商信令算法T-APP）
        # POP_YEAR_INDEX = 154510

        # 常住人口（月度）（运营商信令算法T-APP）
        PER_CAPITA_POP_MONTH_INDEX = 65583

        # Per capita
        pops = input_df[str(PER_CAPITA_POP_MONTH_INDEX)]

        def consider_value_ontology(series: pd.Series) -> pd.Series:
            name = DAAS_INDEX_LIST["index_name"][int(series.name)]
            unit = DAAS_INDEX_LIST["unit"][int(series.name)]
            value_ontology = calc_value_ontology(name, unit)
            if VALUE_ONTOLOGY_CAN_BE_DIVIDED_BY_POPULATION[value_ontology]:
                return series / pops
            return series

        divided_by_pop = input_df.apply(consider_value_ontology)
        output_df = divided_by_pop

        output_instance = DataFrameFile.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=AppendInfoToFeaturesPairwiseFlat)
def create_append_info_to_features_pairwise_flat_job(
    sender: Any,
    instance: AppendInfoToFeaturesPairwiseFlat,
    created: bool,
    **kwargs: Any,
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        input_instance: FeaturesPairwiseFlat = instance.input
        input_df = pd.read_csv(
            input_instance.cached_file.path,
            names=[1, 2, "value"],
            header=0,
        )

        # Deal with comments after index_id
        if instance.has_input_comments:
            input_df[["comment1", "comment2"]] = input_df[[1, 2]].apply(
                lambda s: s.replace("^.*---", "", regex=True)
            )
            input_df[[1, 2]] = input_df[[1, 2]].apply(
                lambda s: s.replace("---.*$", "", regex=True)
            )

        def to_numeric_wrapper(series: pd.Series, errors: str) -> pd.Series:
            return (
                pd.to_numeric(series, errors)
                if str(series.name)[:7] != "comment"
                else series
            )

        input_df = input_df.apply(to_numeric_wrapper, errors="coerce").dropna()

        # Additional info
        input_vo = instance.input_value_ontologies
        if input_vo:
            input_vo_df = pd.read_csv(
                input_vo.cached_file.path,
                names=["index_id", "value_ontology"],
                header=0,
                # index_col=0,
            )
            input_vo_df["index_id"] = input_vo_df["index_id"].astype(int)
            # input_vo_df = input_vo_df.apply(pd.to_numeric, errors="coerce").dropna()

        input_dispersion = instance.input_dispersion
        if input_dispersion:
            input_dispersion_df = pd.read_csv(
                input_dispersion.cached_file.path,
                names=["index_id", "dispersion"],
                header=0,
                # index_col=0,
            )
            input_dispersion_df = input_dispersion_df.apply(
                pd.to_numeric, errors="coerce"
            ).dropna()

        # Joins
        def merge_once(left: pd.DataFrame, index_num: int) -> pd.DataFrame:
            merged_existing = left.rename(columns={index_num: "index_id"})

            KEEP_THESE = [
                "index_name",
                "year_period",
                "type",
                "sub_type",
            ]
            merged_existing = pd.merge(
                merged_existing,
                DAAS_INDEX_LIST[KEEP_THESE],
                on="index_id",
            ).rename(columns={k: f"{k}{index_num}" for k in KEEP_THESE})

            # Append additional info
            if input_vo:
                merged_existing = pd.merge(
                    merged_existing,
                    input_vo_df,
                    on="index_id",
                ).rename(columns={"value_ontology": f"value_ontology{index_num}"})

            if input_dispersion:
                merged_existing = pd.merge(
                    merged_existing,
                    input_dispersion_df,
                    on="index_id",
                ).rename(columns={"dispersion": f"dispersion{index_num}"})

            merged_existing.rename(
                columns={"index_id": f"index_id{index_num}"}, inplace=True
            )
            return merged_existing

        output_df = merge_once(merge_once(input_df, 1), 2)

        output_instance = DataFrameFile.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=DownloadOneMonth)
@on_transaction_commit
def create_download_one_month_job(
    sender: Any, instance: DownloadOneMonth, created: bool, **kwargs: Any
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        month = instance.month

        # Download

        # output_df = get_values_one_month(
        #     month.month_int,
        #     get_gov_level_ids(GovLevel.ALL),
        #     DAAS_INDEX_LIST.index.to_list(),
        # )

        # pylint: disable=fixme
        # Todo: Fix code duplication

        month_int = month.month_int
        gov_ids = get_gov_level_ids(GovLevel.ALL)
        index_ids = DAAS_INDEX_LIST.index.to_list()

        to_be_saved = pd.DataFrame(
            {"gov_id": pd.Series([], dtype="int"), "month": pd.Series([], dtype="int")}
        )

        for i, gov_id in enumerate(gov_ids):
            print(f"Progress: {i + 1} / {len(gov_ids)}")

            instance.progress = int((i + 1) / len(gov_ids) * 100)
            instance.save()

            row = {
                "gov_id": gov_id,
                "month": month_int,
            }

            for trial in range(10):
                try:
                    api_ret = zk2861api.api_get_gov_indexs_data(
                        SECURE_KEY,
                        gov_id,
                        index_ids,
                        f"{month_int_to_str(month_int)}-01",
                        timeout=60,
                    )
                    data = api_ret["data"]
                    for col in data:
                        row[col["index_id"]] = col["value"]
                    to_be_saved = to_be_saved.append(row, ignore_index=True)
                    break
                except KeyError:
                    if trial == 9:
                        return

        output_df = to_be_saved

        # Save

        output_instance = OneMonthGovsAndIndexes.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
            month=month,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


# @receiver(post_save, sender=TimeSeriesSimilarities)
# @on_transaction_commit

# pylint: disable=no-member
@receiver(m2m_changed, sender=TimeSeriesSimilarities.input_months.through)
def create_time_series_similarities_job_set_input_months(
    sender: Any, instance: TimeSeriesSimilarities, **kwargs: Any
) -> None:
    action = kwargs.pop("action")
    empty_input_months = not instance.input_months.exists()

    if "remove" in action:
        raise ValueError("You shouldn't remove input_months.")
    if action == "pre_add" and not empty_input_months:
        raise ValueError("You shouldn't add input_months.")
    if action == "post_add":
        return

    if action == "pre_add" and empty_input_months:
        pk_set = kwargs.pop("pk_set")
        list_data_frames = []
        for key in pk_set:
            input_month = OneMonthGovsAndIndexes.objects.get(pk=key)
            cached_file = input_month.cached_file
            data_frame = pd.read_csv(cached_file.path, index_col=0, nrows=1)
            list_data_frames.append(data_frame)
        concatenated = pd.concat(list_data_frames)

        # Time series
        normalized = (concatenated - concatenated.mean()) / concatenated.std()
        normalized["month"] = concatenated["month"]

        transposed = normalized.transpose()
        formatted = transposed.rename(index={"month": "index_id"})
        formatted.columns = formatted.loc["index_id"]
        final_data = formatted.iloc[2:]

        distances = pdist(final_data.values, metric="euclidean")
        dist_matrix = squareform(distances)

        output_df = pd.DataFrame(
            dist_matrix, index=final_data.index, columns=final_data.index
        )

        output_instance = FeaturesPairwise.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=EachColRelativeOrdering)
def create_each_col_relative_ordering_job(
    sender: Any, instance: EachColRelativeOrdering, created: bool, **kwargs: Any
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        input_instance: DataFrameFile = instance.input
        input_df = pd.read_csv(input_instance.cached_file.path, index_col=0)

        method_of_relative_ordering = instance.method_of_relative_ordering
        if method_of_relative_ordering == EachColRelativeOrdering.PER:
            func = calc_relative_positions
        if method_of_relative_ordering == EachColRelativeOrdering.ORD:
            func = sort_series_and_calc_ordinal_numbers

        # Relative ordering
        output_df = input_df.apply(func)

        output_instance = DataFrameFile.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=DistanceBetweenColPairs)
def create_distance_between_col_pairs_job(
    sender: Any, instance: DistanceBetweenColPairs, created: bool, **kwargs: Any
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        input_instance: DataFrameFile = instance.input
        input_df = pd.read_csv(input_instance.cached_file.path, index_col=0)

        metric = instance.metric

        # Distance
        distances = pdist(input_df.transpose().values, metric=metric)
        dist_matrix = squareform(distances)

        output_df = pd.DataFrame(
            dist_matrix, index=input_df.columns, columns=input_df.columns
        )

        output_instance = FeaturesPairwise.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=LowQualityRowsAndColsRemoval)
def create_low_quality_rows_and_cols_removal_job(
    sender: Any, instance: LowQualityRowsAndColsRemoval, created: bool, **kwargs: Any
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False

        input_df = (
            instance.in_memory_inputs["input"]
            if "input" in instance.in_memory_inputs
            else pd.read_csv(instance.input.cached_file.path, index_col=0)
        )

        min_quality_for_row = instance.min_quality_for_row
        min_quality_for_col = instance.min_quality_for_col

        # Removal

        output_df = input_df.loc[
            input_df.count(1) >= input_df.shape[1] * min_quality_for_row,
            input_df.count(0) >= input_df.shape[0] * min_quality_for_col,
        ]

        if instance.should_output_in_memory:
            instance.in_memory_outputs["output"] = output_df

        instance.output = (
            DataFrameFile.objects.create()
            if not instance.should_output_in_file
            else DataFrameFile.objects.create_data_frame_file(
                data_frame=output_df,
                # predecessor_data_frame_file=input_instance,
                job=instance,
            )
        )

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=FillNA)
def create_fill_na_job(
    sender: Any, instance: FillNA, created: bool, **kwargs: Any
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        input_instance: DataFrameFile = instance.input
        input_df = pd.read_csv(input_instance.cached_file.path, index_col=0)

        # Fill

        output_df = input_df.fillna(input_df.mean())

        output_instance = DataFrameFile.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=CalcValueOntologies)
def create_calc_value_ontologies_job(
    sender: Any, instance: CalcValueOntologies, created: bool, **kwargs: Any
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        # Value ontology
        names_and_units = DAAS_INDEX_LIST[["index_name", "unit"]]

        def calc_value_ontology_wrapper(name_and_unit: pd.Series) -> ValueOntology:
            ontology = calc_value_ontology(
                name_and_unit["index_name"], name_and_unit["unit"]
            )
            return ontology

        value_ontologies = names_and_units.apply(calc_value_ontology_wrapper, axis=1)

        output_df = pd.DataFrame(value_ontologies, index=names_and_units.index)

        output_instance = DataFrameFile.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


# pylint: disable=no-member
@receiver(m2m_changed, sender=DownloadOneGovAcrossTime.months.through)
def create_download_one_gov_across_time_job_set_months(
    sender: Any, instance: DownloadOneGovAcrossTime, **kwargs: Any
) -> None:
    action = kwargs.pop("action")
    empty_months = not instance.months.exists()

    if "remove" in action:
        raise ValueError("You shouldn't remove months.")
    if action == "pre_add" and not empty_months:
        raise ValueError("You shouldn't add months.")
    if action == "post_add":
        return

    if action == "pre_add" and empty_months:
        # pylint: disable=fixme
        # Todo: Fix code duplication

        pk_set = kwargs.pop("pk_set")
        month_ints = [Month.objects.get(pk=key).month_int for key in pk_set]

        gov_id = instance.gov_id
        index_ids = DAAS_INDEX_LIST.index.to_list()

        to_be_saved = pd.DataFrame(
            {"gov_id": pd.Series([], dtype="int"), "month": pd.Series([], dtype="int")}
        )

        for i, month_int in enumerate(sorted(month_ints)):
            print(f"Progress: ({month_int}) {i + 1} / {len(pk_set)}")

            instance.progress = int((i + 1) / len(pk_set) * 100)
            instance.save()

            row = {
                "gov_id": gov_id,
                "month": month_int,
            }

            for trial in range(10):
                try:
                    api_ret = zk2861api.api_get_gov_indexs_data(
                        SECURE_KEY,
                        gov_id,
                        index_ids,
                        f"{month_int_to_str(month_int)}-01",
                        timeout=60,
                    )
                    data = api_ret["data"]
                    for col in data:
                        row[col["index_id"]] = col["value"]
                    to_be_saved = to_be_saved.append(row, ignore_index=True)
                    break
                except KeyError:
                    if trial == 9:
                        return

        output_df = to_be_saved

        output_instance = DataFrameFile.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=FilterByInfo)
def create_filter_by_info_job(
    sender: Any, instance: FilterByInfo, created: bool, **kwargs: Any
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        input_instance: DataFrameFile = instance.input
        input_df = pd.read_csv(input_instance.cached_file.path, index_col=0)

        no_same_type = instance.no_same_type
        no_year_period = instance.no_year_period
        min_dispersion = instance.min_dispersion

        # Filter
        output_df = input_df
        if no_same_type:
            output_df = output_df.loc[~(output_df["type1"] == output_df["type2"]), :]
        if no_year_period:
            output_df = output_df.loc[
                ~(output_df["year_period1"] == "t")
                & ~(output_df["year_period2"] == "t"),
                :,
            ]
        if min_dispersion is not None:
            output_df = output_df.loc[
                (output_df["dispersion1"] > min_dispersion)
                & (output_df["dispersion2"] > min_dispersion),
                :,
            ]

        output_instance = DataFrameFile.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=ConcatDataFrames)
def create_concat_data_frames_job(
    sender: Any, instance: ConcatDataFrames, created: bool, **kwargs: Any
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        input1: DataFrameFile = instance.input1
        input2: DataFrameFile = instance.input2
        input1_df = pd.read_csv(input1.cached_file.path, index_col=0)
        input2_df = pd.read_csv(input2.cached_file.path, index_col=0)

        comment = instance.comment

        # Filter
        with_comment = input2_df.rename(
            columns={column: f"{column}---{comment}" for column in input2_df.columns}
        )
        output_df = pd.concat([input1_df, with_comment], axis=1)

        output_instance = DataFrameFile.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


# pylint: disable=no-member
@receiver(m2m_changed, sender=DownloadOneGovMacroDataAcrossTime.months.through)
def create_download_one_gov_macro_data_across_time_job_set_months(
    sender: Any, instance: DownloadOneGovMacroDataAcrossTime, **kwargs: Any
) -> None:
    action = kwargs.pop("action")
    empty_months = not instance.months.exists()

    if "remove" in action:
        raise ValueError("You shouldn't remove months.")
    if action == "pre_add" and not empty_months:
        raise ValueError("You shouldn't add months.")
    if action == "post_add":
        return

    if action == "pre_add" and empty_months:
        # pylint: disable=fixme
        # Todo: Fix code duplication

        pk_set = kwargs.pop("pk_set")
        month_ints = [Month.objects.get(pk=key).month_int for key in pk_set]

        gov_id = instance.gov_id
        node_ids = MACRO_DATA_DIR_INDEX_NODES.index.to_list()

        to_be_saved = pd.DataFrame(
            {"gov_id": pd.Series([], dtype="int"), "month": pd.Series([], dtype="int")}
        )

        for i, month_int in enumerate(sorted(month_ints)):
            print(f"Progress: ({month_int}) {i + 1} / {len(pk_set)}")

            instance.progress = int((i + 1) / len(pk_set) * 100)
            instance.save()

            row = {
                "gov_id": gov_id,
                "month": month_int,
            }

            for trial in range(10):
                try:
                    res_df = api_get_macro_data_gov_nodes(
                        gov_id, node_ids, f"{month_int_to_str(month_int)}-01"
                    )
                    for _, res_row in res_df.iterrows():
                        row[res_row["type_code"]] = res_row["node_sum"]
                    to_be_saved = to_be_saved.append(row, ignore_index=True)
                    break
                except KeyError:
                    if trial == 9:
                        return

        output_df = to_be_saved

        output_instance = DataFrameFile.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=AppendMacroDataInfoToFeaturesPairwiseFlat)
def create_append_macro_data_info_to_features_pairwise_flat_job(
    sender: Any,
    instance: AppendMacroDataInfoToFeaturesPairwiseFlat,
    created: bool,
    **kwargs: Any,
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        input_instance: FeaturesPairwiseFlat = instance.input
        input_df = pd.read_csv(
            input_instance.cached_file.path,
            names=[1, 2, "value"],
            header=0,
        )

        # Deal with comments after node_id
        if instance.has_input_comments:
            input_df[["comment1", "comment2"]] = input_df[[1, 2]].apply(
                lambda s: s.replace("^.*---", "", regex=True)
            )
            input_df[[1, 2]] = input_df[[1, 2]].apply(
                lambda s: s.replace("---.*$", "", regex=True)
            )

        # def to_numeric_wrapper(series: pd.Series, errors: str) -> pd.Series:
        #     return (
        #         pd.to_numeric(series, errors)
        #         if str(series.name)[:7] != "comment"
        #         else series
        #     )

        # input_df = input_df.apply(to_numeric_wrapper, errors="coerce").dropna()

        # Additional info
        input_dispersion = instance.input_dispersion
        if input_dispersion:
            input_dispersion_df = pd.read_csv(
                input_dispersion.cached_file.path,
                names=["node_id", "dispersion"],
                header=0,
                # index_col=0,
            )
            # input_dispersion_df = input_dispersion_df.apply(
            #     pd.to_numeric, errors="coerce"
            # ).dropna()

        # Joins
        def merge_once(left: pd.DataFrame, index_num: int) -> pd.DataFrame:
            merged_existing = left.rename(columns={index_num: "node_id"})

            KEEP_THESE = [
                "node_name",
                "year_period",
                "unit",
            ]
            merged_existing = pd.merge(
                merged_existing,
                MACRO_DATA_DIR_INDEX_NODES[KEEP_THESE],
                on="node_id",
            ).rename(columns={k: f"{k}{index_num}" for k in KEEP_THESE})

            # Append additional info
            if input_dispersion:
                merged_existing = pd.merge(
                    merged_existing,
                    input_dispersion_df,
                    on="node_id",
                ).rename(columns={"dispersion": f"dispersion{index_num}"})

            merged_existing.rename(
                columns={"node_id": f"node_id{index_num}"}, inplace=True
            )
            return merged_existing

        output_df = merge_once(merge_once(input_df, 1), 2)

        output_instance = DataFrameFile.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=DiscretizeEachCol)
def create_discretize_each_col_job(
    sender: Any, instance: DiscretizeEachCol, created: bool, **kwargs: Any
) -> None:
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False
        input_instance: DataFrameFile = instance.input
        input_df = pd.read_csv(input_instance.cached_file.path, index_col=0)

        number_of_bins = instance.number_of_bins

        # Discretize
        output_df = input_df.apply(
            lambda s: pd.qcut(s, number_of_bins, labels=False, duplicates="drop")
        )

        output_instance = DataFrameFile.objects.create_data_frame_file(
            data_frame=output_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()


@receiver(post_save, sender=AppendInfoToAssociationRules)
def create_append_info_to_association_rules_job(
    sender: Any, instance: AppendInfoToAssociationRules, created: bool, **kwargs: Any
) -> None:
    """
    Todo: Implement this function as a capital job
    """
    if getattr(instance, "my_manager_created", False):
        instance.my_manager_created = False

        input_instance: DataFrameFile = instance.input
        input_df = pd.read_csv(input_instance.cached_file.path, index_col=0)

        number_of_bins = instance.number_of_bins

        # Deal with (antecedents and consequents) or itemsets
        TO_BE_DEALT_WITH = [
            i
            for i in ["antecedents", "consequents", "itemsets"]
            if i in input_df.columns
        ]

        def deal_with_ante_and_cons_or_itemsets(frozen_set: str) -> list[str]:
            stripped = frozen_set.strip(r"abcdefghijklmnopqrstuvwxyz(){}")
            splitted = [item.strip("'") for item in stripped.split(", ")]
            return splitted

        for to_be_dealt_with in TO_BE_DEALT_WITH:
            input_df[to_be_dealt_with] = input_df[to_be_dealt_with].apply(
                deal_with_ante_and_cons_or_itemsets
            )

        # Additional info
        input_vo = instance.input_value_ontologies
        if input_vo:
            input_vo_df = pd.read_csv(
                input_vo.cached_file.path,
                names=["index_id", "value_ontology"],
                header=0,
                # index_col=0,
            )
            input_vo_df["index_id"] = input_vo_df["index_id"].astype(int)
            input_vo_df.set_index("index_id", inplace=True)

        input_dispersion = instance.input_dispersion
        if input_dispersion:
            input_dispersion_df = pd.read_csv(
                input_dispersion.cached_file.path,
                names=["index_id", "dispersion"],
                header=0,
                # index_col=0,
            )
            input_dispersion_df = input_dispersion_df.apply(
                pd.to_numeric, errors="coerce"
            ).dropna()
            input_dispersion_df.set_index("index_id", inplace=True)

        # Joins
        def merge_once(series: pd.Series) -> None:
            KEEP_THESE = [
                DAAS_INDEX_LIST["index_name"],
                DAAS_INDEX_LIST["year_period"],
                DAAS_INDEX_LIST["type"],
                DAAS_INDEX_LIST["sub_type"],
                input_vo_df["value_ontology"],
                input_dispersion_df["dispersion"],
            ]
            APPEND_GOOD_COMMENTS_AFTER_THESE = [
                "index_name",
            ]
            GOOD_COMMENTS = {
                1: {"0": "all"},
                2: {"0": "low", "1": "high"},
                3: {"0": "low", "1": "mid", "2": "high"},
                4: {"0": "low", "1": "mid-low", "2": "mid-high", "3": "high"},
            }

            def retrieve(index_id_with_comment: str, field_col: pd.Series) -> str:
                index_id, comment = index_id_with_comment.split("---")
                retrieved = (
                    field_col[int(index_id)] if index_id.isdecimal() else index_id
                )
                return (
                    f"{retrieved} [{GOOD_COMMENTS[number_of_bins][comment]}]"
                    if field_col.name in APPEND_GOOD_COMMENTS_AFTER_THESE
                    else retrieved
                )

            for field in KEEP_THESE:
                input_df[f"{series.name} {field.name}"] = series.apply(
                    lambda x: [
                        retrieve(index_id_with_comment, field)
                        for index_id_with_comment in x
                    ]
                )

        for to_be_dealt_with in TO_BE_DEALT_WITH:
            merge_once(input_df[to_be_dealt_with])

        output_instance = DataFrameFile.objects.create_data_frame_file(
            data_frame=input_df,
            job=instance,
        )

        instance.output = output_instance

        instance.progress = 100
        instance.save()
