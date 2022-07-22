import json
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from dtw import dtw
from sklearn import cluster

from base.models import (
    SEVEN_AREAS_CHOICES,
    AssociationRules,
    ByIndustryFiltering,
    DiscretizeAndToOneHot,
    DistrictsIncomeLevelFiltering,
    DistrictsPopulationLevelFiltering,
    DistrictsSevenAreasLevelFiltering,
    DynamicTimeWarping,
    FilterByValueInOneCol,
    FrequentItemSets,
    KMeans,
    NormalizeEachCol,
    ProvincialCapitalsChildrenFiltering,
    RemoveDispersionSmallCol,
    RemoveIndexesThatHaveSomeStringInNames,
    RemoveTooManyDuplicatesCol,
)
from utils import (
    DAAS_INDEX_LIST,
    INCOME_PER_CAPITA_INDEX,
    POP_MONTH_INDEX,
    DistrictsIncomeLevel,
    DistrictsPopulationLevel,
    calc_dispersions,
    calc_district_income_level,
    calc_district_population_level,
    calc_district_seven_areas_level,
)
from utils.provincial_capitals import PROVINCIAL_CAPITALS_CHILDREN


def association_rules_work(
    input_df: pd.DataFrame, job: AssociationRules
) -> pd.DataFrame:
    max_len = job.max_len
    min_support = job.min_support
    METRIC = "confidence"
    MIN_THRESHOLD = 0.8
    try:
        apriori_results = apriori(
            input_df,
            use_colnames=True,
            min_support=min_support,
            max_len=max_len,
            low_memory=False,
            verbose=1,
        )
    except MemoryError:
        print("Using fpgrowth")
        apriori_results = fpgrowth(
            input_df,
            use_colnames=True,
            min_support=min_support,
            max_len=max_len,
            verbose=1,
            # low_memory=True,
        )
    df_ar = association_rules(
        apriori_results, metric=METRIC, min_threshold=MIN_THRESHOLD
    )
    return df_ar


def dynamic_time_warping_work(
    input_df: pd.DataFrame, job: DynamicTimeWarping
) -> pd.DataFrame:
    keep_track: dict[pd.Series, dict[pd.Series, float]] = {}
    keep_track_count = 0

    def dtw_wrapper(
        series: pd.Series,
        series2: pd.Series,
    ) -> float:
        if series2.name in keep_track and series.name in keep_track[series2.name]:
            return keep_track[series2.name][series.name]

        result = dtw(series, series2, lambda x, y: np.abs(x - y))[0]

        keep_track[series.name] = (
            keep_track[series.name] if series.name in keep_track else {}
        )
        keep_track[series.name][series2.name] = result
        nonlocal keep_track_count
        keep_track_count += 1

        # Progress
        total_steps = len(input_df.columns) * (len(input_df.columns) + 1) / 2
        print(f"Progress: {keep_track_count} / {total_steps}")
        job.progress = int(keep_track_count / total_steps * 100)
        job.save()

        return result

    output_df = input_df.apply(
        lambda series: pd.Series(
            [dtw_wrapper(series, series2) for _, series2 in input_df.iteritems()],
            input_df.columns,
        )
    )
    return output_df


def normalize_each_col_work(
    input_df: pd.DataFrame, job: NormalizeEachCol
) -> pd.DataFrame:
    normalized = (input_df - input_df.mean()) / input_df.std()
    return normalized


def discretize_and_to_one_hot_work(
    input_df: pd.DataFrame, job: DiscretizeAndToOneHot
) -> pd.DataFrame:
    number_of_bins = job.number_of_bins

    discretized = input_df.apply(
        lambda s: pd.qcut(s, number_of_bins, labels=False, duplicates="drop")
    )

    # One hot
    def rename_columns(series: pd.Series) -> dict[int, str]:
        return {i: f"{series.name}---{i}" for i in range(number_of_bins)}

    all_dummies = []
    discretized.apply(
        lambda series: all_dummies.append(
            pd.get_dummies(series).rename(columns=rename_columns(series))
        )
    )

    one_hot = pd.concat(all_dummies, axis=1)

    return one_hot


def remove_dispersion_small_col_work(
    input_df: pd.DataFrame, job: RemoveDispersionSmallCol
) -> pd.DataFrame:
    measure_of_dispersion = job.measure_of_dispersion
    min_dispersion = job.min_dispersion

    dispersions = calc_dispersions(input_df, measure_of_dispersion)

    output_df = input_df.loc[
        :, [dispersions[col] > min_dispersion for col in input_df.columns]
    ]

    return output_df


def remove_too_many_duplicates_col_work(
    input_df: pd.DataFrame, job: RemoveTooManyDuplicatesCol
) -> pd.DataFrame:
    mdp = job.max_duplicates_percentage
    output_df = input_df.loc[
        :,
        [
            sum(input_df[col].duplicated()) / len(input_df[col]) < mdp
            for col in input_df.columns
        ],
    ]
    return output_df


def frequent_item_sets_work(
    input_df: pd.DataFrame, job: FrequentItemSets
) -> pd.DataFrame:
    max_len = job.max_len
    min_support = job.min_support
    try:
        apriori_results = apriori(
            input_df,
            use_colnames=True,
            min_support=min_support,
            max_len=max_len,
            low_memory=False,
            verbose=1,
        )
    except MemoryError:
        print("Using fpgrowth")
        apriori_results = fpgrowth(
            input_df,
            use_colnames=True,
            min_support=min_support,
            max_len=max_len,
            verbose=1,
            # low_memory=True,
        )
    return apriori_results


def remove_indexes_that_have_some_string_in_names(
    input_df: pd.DataFrame, job: RemoveIndexesThatHaveSomeStringInNames
) -> pd.DataFrame:
    some_string = job.some_string
    index_names = DAAS_INDEX_LIST["index_name"]
    output_df = input_df.loc[
        :,
        [
            col
            for col in input_df.columns
            if (isinstance(col, int) or col.isdecimal())
            and int(col) in index_names
            and some_string not in index_names[int(col)]
        ],
    ]
    return output_df


def k_means_work(input_df: pd.DataFrame, job: KMeans) -> pd.DataFrame:
    n_clusters = job.n_clusters
    use_cols = job.use_cols

    dropped_na = (
        input_df.dropna(axis=1)
        if not use_cols
        else input_df[json.loads(use_cols)].dropna(axis=0)
    )
    labels = cluster.KMeans(n_clusters=n_clusters).fit_predict(dropped_na)
    labels_df = pd.DataFrame(labels, index=dropped_na.index).rename(
        columns={0: "k_means_label"}
    )
    output_df = pd.concat([dropped_na, labels_df], axis=1)
    return output_df


def filter_by_value_in_one_col_work(
    input_df: pd.DataFrame, job: FilterByValueInOneCol
) -> pd.DataFrame:
    use_col = job.use_col
    min_value = job.min_value if job.min_value is not None else -np.inf
    max_value = job.max_value if job.max_value is not None else np.inf

    output_df = input_df.loc[
        (input_df[use_col] >= min_value) & (input_df[use_col] <= max_value), :
    ]
    return output_df


def districts_population_level_filtering_work(
    input_df: pd.DataFrame, job: DistrictsPopulationLevelFiltering
) -> pd.DataFrame:
    pop_level = job.pop_level

    filtered = input_df[
        input_df[str(POP_MONTH_INDEX)].apply(
            lambda pop: calc_district_population_level(pop)
            == DistrictsPopulationLevel(pop_level)
        )
    ]
    filtered.reset_index(inplace=True, drop=True)
    return filtered


def districts_income_level_filtering_work(
    input_df: pd.DataFrame, job: DistrictsIncomeLevelFiltering
) -> pd.DataFrame:
    income_level = job.income_level

    filtered = input_df[
        input_df[str(INCOME_PER_CAPITA_INDEX)].apply(
            lambda income: calc_district_income_level(income)
            == DistrictsIncomeLevel(income_level)
        )
    ]
    filtered.reset_index(inplace=True, drop=True)
    return filtered


def districts_seven_areas_level_filtering_work(
    input_df: pd.DataFrame, job: DistrictsSevenAreasLevelFiltering
) -> pd.DataFrame:
    seven_areas_level = job.seven_areas_level
    seven_areas_level_str = [
        choice[1] for choice in SEVEN_AREAS_CHOICES if choice[0] == seven_areas_level
    ][0]

    filtered = input_df[
        input_df["gov_id"].apply(
            lambda gov_id: calc_district_seven_areas_level(gov_id)
            == seven_areas_level_str
        )
    ]
    filtered.reset_index(inplace=True, drop=True)
    return filtered


def provincial_capitals_children_filtering_work(
    input_df: pd.DataFrame, job: ProvincialCapitalsChildrenFiltering
) -> pd.DataFrame:
    provincial_capitals_children = job.provincial_capitals_children
    provincial_capitals_children_ids = [
        child["gov_id"] for child in PROVINCIAL_CAPITALS_CHILDREN
    ]

    filtered = (
        input_df.loc[input_df["gov_id"].isin(provincial_capitals_children_ids)]
        if provincial_capitals_children
        else input_df.loc[~input_df["gov_id"].isin(provincial_capitals_children_ids)]
    )
    filtered.reset_index(inplace=True, drop=True)
    return filtered


def by_industry_filtering_work(
    input_df: pd.DataFrame, job: ByIndustryFiltering
) -> pd.DataFrame:
    industry = job.industry
    FIRST_N_ROWS = 100

    filtered = input_df.sort_values(str(industry), ascending=False).head(FIRST_N_ROWS)
    filtered.reset_index(inplace=True, drop=True)
    return filtered
