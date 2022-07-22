from typing import Any, Callable, Type
from django.db.models import Model
from django.db.models.signals import ModelSignal
from django.dispatch import receiver
from django.db import transaction
from django.db.models.signals import post_save, m2m_changed

import pandas as pd
from base.jobs.capital_jobs_works import (
    association_rules_work,
    by_industry_filtering_work,
    discretize_and_to_one_hot_work,
    districts_income_level_filtering_work,
    districts_population_level_filtering_work,
    districts_seven_areas_level_filtering_work,
    dynamic_time_warping_work,
    filter_by_value_in_one_col_work,
    frequent_item_sets_work,
    k_means_work,
    normalize_each_col_work,
    provincial_capitals_children_filtering_work,
    remove_dispersion_small_col_work,
    remove_indexes_that_have_some_string_in_names,
    remove_too_many_duplicates_col_work,
)

from base.models import (
    AssociationRules,
    ByIndustryFiltering,
    DataFrameFile,
    DiscretizeAndToOneHot,
    DistrictsIncomeLevelFiltering,
    DistrictsPopulationLevelFiltering,
    DistrictsSevenAreasLevelFiltering,
    DynamicTimeWarping,
    FeaturesPairwise,
    FilterByValueInOneCol,
    FrequentItemSets,
    Job,
    KMeans,
    NormalizeEachCol,
    ProvincialCapitalsChildrenFiltering,
    RemoveDispersionSmallCol,
    RemoveIndexesThatHaveSomeStringInNames,
    RemoveTooManyDuplicatesCol,
)


CONFIGS = (
    (
        AssociationRules,
        post_save,
        DataFrameFile,
        association_rules_work,
    ),
    (
        DynamicTimeWarping,
        post_save,
        FeaturesPairwise,
        dynamic_time_warping_work,
    ),
    (
        NormalizeEachCol,
        post_save,
        DataFrameFile,
        normalize_each_col_work,
    ),
    (
        DiscretizeAndToOneHot,
        post_save,
        DataFrameFile,
        discretize_and_to_one_hot_work,
    ),
    (
        RemoveDispersionSmallCol,
        post_save,
        DataFrameFile,
        remove_dispersion_small_col_work,
    ),
    (
        RemoveTooManyDuplicatesCol,
        post_save,
        DataFrameFile,
        remove_too_many_duplicates_col_work,
    ),
    (
        FrequentItemSets,
        post_save,
        DataFrameFile,
        frequent_item_sets_work,
    ),
    (
        RemoveIndexesThatHaveSomeStringInNames,
        post_save,
        DataFrameFile,
        remove_indexes_that_have_some_string_in_names,
    ),
    (
        KMeans,
        post_save,
        DataFrameFile,
        k_means_work,
    ),
    (
        FilterByValueInOneCol,
        post_save,
        DataFrameFile,
        filter_by_value_in_one_col_work,
    ),
    (
        DistrictsPopulationLevelFiltering,
        post_save,
        DataFrameFile,
        districts_population_level_filtering_work,
    ),
    (
        DistrictsIncomeLevelFiltering,
        post_save,
        DataFrameFile,
        districts_income_level_filtering_work,
    ),
    (
        DistrictsSevenAreasLevelFiltering,
        post_save,
        DataFrameFile,
        districts_seven_areas_level_filtering_work,
    ),
    (
        ProvincialCapitalsChildrenFiltering,
        post_save,
        DataFrameFile,
        provincial_capitals_children_filtering_work,
    ),
    (
        ByIndustryFiltering,
        post_save,
        DataFrameFile,
        by_industry_filtering_work,
    ),
)


# Higher order functions


def on_transaction_commit_decorator(on_transaction_commit: bool) -> Callable:
    def on_transaction_commit_true(func: Callable) -> Callable:
        def what_to_do(*args: Any, **kwargs: Any) -> Any:
            # print("committed!")
            return func(*args, **kwargs)

        return lambda *args, **kwargs: transaction.on_commit(
            lambda: what_to_do(*args, **kwargs)
        )

    def on_transaction_commit_false(func: Callable) -> Callable:
        return func

    return (
        on_transaction_commit_true
        if on_transaction_commit
        else on_transaction_commit_false
    )


def make_signal(
    sender: Type[Job] | Type[Model],
    signal: ModelSignal,
    output_type: Type[DataFrameFile],
    work: Callable[[pd.DataFrame, Job], pd.DataFrame],
    on_transaction_commit: bool = True,
) -> Callable:
    @receiver(signal, sender=sender)
    @on_transaction_commit_decorator(on_transaction_commit)
    def _receiver(sender: Any, instance: Job, created: bool, **kwargs: Any) -> None:

        # Only do when created by my manager
        if getattr(instance, "my_manager_created", False):
            instance.my_manager_created = False

            # Input data frame
            input_df = (
                instance.in_memory_inputs["input"]
                if "input" in instance.in_memory_inputs
                else pd.read_csv(instance.input.cached_file.path, index_col=0)
            )

            # Do work
            output_df = work(input_df, instance)

            # If should output in memory
            if instance.should_output_in_memory:
                instance.in_memory_outputs["output"] = output_df

            # Save instance
            instance.output = (
                output_type.objects.create_data_frame_file(
                    data_frame=output_df,
                    # predecessor_data_frame_file=input_instance,
                    job=instance,
                )
                if instance.should_output_in_file
                else output_type.objects.create()
            )

            # Progress
            instance.progress = 100
            instance.save()

    return _receiver


# Making


for config in CONFIGS:
    make_signal(*config)
