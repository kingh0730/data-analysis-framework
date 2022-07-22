from typing import Any, Callable
import pandas as pd
from base.models import Job, Month
from techniques.technique_exec import MiningTechnique, SampleSelectionTechnique

from utils import (
    GOVS_INFO,
    POP_MONTH_INDEX,
    DistrictsIncomeLevel,
    DistrictsPopulationLevel,
    GovLevel,
    MeasureOfDispersion,
    calc_district_population_level,
    correlation_interpretation,
    frequent_item_sets_interpretation,
    sample_selection_districts_by_industry_interpretation,
    sample_selection_districts_income_interpretation,
    sample_selection_districts_interpretation,
    sample_selection_districts_population_interpretation,
    sample_selection_districts_provincial_capitals_children_interpretation,
    sample_selection_districts_seven_areas_interpretation,
    sample_selection_time_series_interpretation,
)


def interpret_sample_selection_districts(
    index_id: int, gov_id: int, samples: pd.DataFrame
) -> dict:
    return {
        "criteria": [
            {
                "name": "gov_level",
                "class": GovLevel.DISTRICT_LEVEL.name,
            }
        ],
        "count": len(samples),
        "interpretation": sample_selection_districts_interpretation(
            gov_id, GOVS_INFO["name"][gov_id], len(samples)
        ),
    }


def interpret_sample_selection_districts_population(
    index_id: int,
    gov_id: int,
    samples: pd.DataFrame,
    pop_level: DistrictsPopulationLevel,
) -> dict:
    return {
        "criteria": [
            {
                "name": "gov_level",
                "class": GovLevel.DISTRICT_LEVEL.name,
            },
            {
                "name": "population",
                "class": pop_level.name,
            },
        ],
        "count": len(samples),
        "interpretation": sample_selection_districts_population_interpretation(
            gov_id, GOVS_INFO["name"][gov_id], len(samples), pop_level
        ),
    }


def interpret_sample_selection_districts_income(
    index_id: int,
    gov_id: int,
    samples: pd.DataFrame,
    income_level: DistrictsIncomeLevel,
) -> dict:
    return {
        "criteria": [
            {
                "name": "gov_level",
                "class": GovLevel.DISTRICT_LEVEL.name,
            },
            {
                "name": "income",
                "class": income_level.name,
            },
        ],
        "count": len(samples),
        "interpretation": sample_selection_districts_income_interpretation(
            gov_id, GOVS_INFO["name"][gov_id], len(samples), income_level
        ),
    }


def interpret_sample_selection_districts_seven_areas(
    index_id: int,
    gov_id: int,
    samples: pd.DataFrame,
    seven_areas_level: Any,
) -> dict:
    return {
        "criteria": [
            {
                "name": "gov_level",
                "class": GovLevel.DISTRICT_LEVEL.name,
            },
            {
                "name": "seven_area",
                "class": seven_areas_level.name,
            },
        ],
        "count": len(samples),
        "interpretation": sample_selection_districts_seven_areas_interpretation(
            gov_id, GOVS_INFO["name"][gov_id], len(samples), seven_areas_level
        ),
    }


def interpret_sample_selection_districts_provincial_capitals_children(
    index_id: int, gov_id: int, samples: pd.DataFrame
) -> dict:
    return {
        "criteria": [
            {
                "name": "gov_level",
                "class": GovLevel.DISTRICT_LEVEL.name,
            },
            {
                "name": "provincial_capitals_children",
                "class": True,
            },
        ],
        "count": len(samples),
        "interpretation": sample_selection_districts_provincial_capitals_children_interpretation(
            gov_id, GOVS_INFO["name"][gov_id], len(samples)
        ),
    }


def interpret_sample_selection_districts_by_industry(
    index_id: int,
    gov_id: int,
    samples: pd.DataFrame,
    industry: Any,
) -> dict:
    return {
        "criteria": [
            {
                "name": "gov_level",
                "class": GovLevel.DISTRICT_LEVEL.name,
            },
            {
                "name": "by_industry",
                "class": industry.name,
            },
        ],
        "count": len(samples),
        "interpretation": sample_selection_districts_by_industry_interpretation(
            gov_id, GOVS_INFO["name"][gov_id], len(samples), industry
        ),
    }


def interpret_sample_selection_time_series(
    index_id: int,
    gov_id_positional: int,
    samples: pd.DataFrame,
    months: Any,
    gov_id: Any,
) -> dict:
    return {
        "criteria": [
            {
                "name": "time_series",
                "class": months.name,
            },
        ],
        "count": len(samples),
        "interpretation": sample_selection_time_series_interpretation(
            gov_id_positional,
            GOVS_INFO["name"][gov_id_positional],
            len(samples),
            months,
        ),
    }


def interpret_pearson_corr(
    index_id: int, gov_id: int, appended_info_df: pd.DataFrame
) -> list[dict]:
    biggest = appended_info_df.loc[
        (appended_info_df["index_id1"] == index_id)
        | (appended_info_df["index_id2"] == index_id),
        :,
    ].sort_values("value", ascending=False)

    if len(biggest) == 0:
        return []

    big = biggest.iloc[0]
    index_id_is_1 = big["index_id1"] == index_id
    other_index_id = big["index_id2"] if index_id_is_1 else big["index_id1"]
    interpretation = correlation_interpretation(
        index_id,
        big["index_name1"] if index_id_is_1 else big["index_name2"],
        other_index_id,
        big["index_name2"] if index_id_is_1 else big["index_name1"],
        big["value"],
    )
    json_obj = [
        {
            "index_id": int(other_index_id),
            "technique": "pearson-corr",
            "values": [
                {
                    "name": "corr",
                    "value": big["value"],
                    "interpretation": interpretation,
                },
            ],
            "byproducts": {
                # "outliers_ids": [1, 5, 88, 234],
            },
            "gov_specific": {
                # "is_outlier": False,
            },
        }
    ]
    return json_obj


def interpret_frequent_item_sets(
    index_id: int, gov_id: int, appended_info_df: pd.DataFrame
) -> list[dict]:
    itemsets_ids = [
        [
            int(quote.strip("'").split("---")[0])
            for quote in itemset.strip("[]").split(", ")
        ]
        for itemset in appended_info_df["itemsets"]
    ]
    biggest = appended_info_df
    biggest["itemsets_ids"] = itemsets_ids
    biggest = appended_info_df.loc[
        [index_id in itemset_ids for itemset_ids in itemsets_ids],
        :,
    ].sort_index(ascending=False)

    if len(biggest) == 0:
        return []

    big = biggest.iloc[0]
    interpretation = frequent_item_sets_interpretation(
        [
            item.strip("'")
            for item in big["itemsets index_name"].strip("[]").split(", ")
        ],
        big["support"],
    )
    json_obj = [
        {
            "index_id": other_index_id,
            "technique": "frequent-item-sets",
            "values": [
                {
                    "name": "support",
                    "value": big["support"],
                    "interpretation": interpretation,
                },
            ],
            "byproducts": {
                # "outliers_ids": [1, 5, 88, 234],
            },
            "gov_specific": {
                # "is_outlier": False,
            },
        }
        for other_index_id in big["itemsets_ids"]
        if other_index_id != index_id
    ]
    return json_obj


# Dicts


SAMPLE_SELECTION_TECHNIQUES_ITP: dict[SampleSelectionTechnique, Callable] = {
    SampleSelectionTechnique.DISTRICTS: interpret_sample_selection_districts,
    SampleSelectionTechnique.DISTRICTS_POPULATION: interpret_sample_selection_districts_population,
    SampleSelectionTechnique.DISTRICTS_INCOME: interpret_sample_selection_districts_income,
    SampleSelectionTechnique.DISTRICTS_SEVEN_AREAS: interpret_sample_selection_districts_seven_areas,
    SampleSelectionTechnique.DISTRICTS_PROVINCIAL_CAPITALS_CHILDREN: interpret_sample_selection_districts_provincial_capitals_children,
    SampleSelectionTechnique.DISTRICTS_BY_INDUSTRY: interpret_sample_selection_districts_by_industry,
    SampleSelectionTechnique.TIME_SERIES: interpret_sample_selection_time_series,
}


MINING_TECHNIQUES_ITP: dict[MiningTechnique, Callable] = {
    MiningTechnique.PEARSON_CORR: interpret_pearson_corr,
    MiningTechnique.FREQUENT_ITEM_SETS: interpret_frequent_item_sets,
    # Todo: Probably it wants its own interpretation.
    MiningTechnique.TIME_SERIES: interpret_pearson_corr,
}


# Aggregate


def aggregate_interpretations(
    samples_itp: dict,
    relations_itp: list[dict],
    has_multiple_govs: bool,
) -> dict:
    samples_itp_agg = samples_itp["interpretation"]
    relations_itp_agg = "；".join(
        list(
            {
                value["interpretation"]
                for itp in relations_itp
                for value in itp["values"]
            }
        )
    )
    link_string = "。通过对这些地区的分析，" if has_multiple_govs else "，"
    relations_itp_agg_sentence = (
        f"{link_string}{relations_itp_agg}。" if relations_itp_agg else ""
    )
    json_obj = {
        "samples": samples_itp,
        "relations": relations_itp,
        "interpretation": f"{samples_itp_agg}{relations_itp_agg_sentence}",
    }
    return json_obj
