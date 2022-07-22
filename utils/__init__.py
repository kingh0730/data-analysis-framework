__version__ = "0.1.0"


import enum
from itertools import chain
from typing import Any, KeysView
import numpy as np
import pandas as pd
from scipy.stats import variation


from . import zk2861api
from . import gov_bar_data

SECURE_KEY = "NDRkOGJlZTFkMDQzMjUwZTQ0ZjIyMmU5YjIwY2U0NDU="
zk2861api.ENCRYPT_ENABLE = 0


def month_int_to_str(month_int: int) -> str:
    month_int = month_int - 1
    year = month_int // 12 + 2001
    month = month_int % 12 + 1
    return f"{year}-{0 if month < 10 else ''}{month}"


def remove_outliers_series_with_median_deviation(
    series: pd.Series, outlier_sd_threshold: float
) -> pd.Series:
    dropped_na = series.dropna()
    deviation = np.abs(dropped_na - np.median(dropped_na))
    median_deviation = np.median(deviation)
    scaled_deviation = (deviation / median_deviation) if median_deviation else None

    result = (
        dropped_na[scaled_deviation < outlier_sd_threshold]
        if scaled_deviation is not None
        else pd.Series(index=series.index, dtype=np.float64)
    )

    return result


def remove_columns_that_have_too_many_nas(
    data_frame: pd.DataFrame, min_count: float
) -> pd.DataFrame:
    valid_columns = [c for c in data_frame if len(data_frame[c].dropna()) >= min_count]
    return data_frame[valid_columns]


def sort_corr(corr: pd.DataFrame) -> pd.DataFrame:
    sorted_corr = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .unstack()
        .dropna()
        .sort_values(ascending=False)
    )

    return sorted_corr


# zk2861api


class GovLevel(enum.Enum):
    ALL = -1
    NATION = 0
    PROVINCE = 1
    CITY_LEVEL = 2
    DISTRICT_LEVEL = 3


def _get_govs_info() -> Any:
    # permit govs
    api_ret = zk2861api.api_get_permit_gov_region_list(SECURE_KEY, timeout=30)
    all_govs_info = {d["gov_id"]: d for d in api_ret["data"]}
    print(f"all_govs length: {len(all_govs_info)}")

    return all_govs_info


def _get_govs_info_as_df() -> pd.DataFrame:
    # permit govs
    api_ret = zk2861api.api_get_permit_gov_region_list(SECURE_KEY, timeout=30)
    all_govs_info = pd.DataFrame(api_ret["data"])
    all_govs_info.set_index("gov_id", inplace=True)
    print(f"all_govs length: {len(all_govs_info)}")

    return all_govs_info


GOVS_INFO = _get_govs_info_as_df()


NATION_INFO = GOVS_INFO[GOVS_INFO["level"] == "全国"]
PROVINCES_INFO = GOVS_INFO[GOVS_INFO["level"] == "省"]

_CITIES_INFO = GOVS_INFO[GOVS_INFO["level"] == "市"]
_MUNICIPALITIES_INFO = GOVS_INFO[GOVS_INFO["level"] == "直辖市"]
_DISTRICTS_INFO = GOVS_INFO[GOVS_INFO["level"] == "区县"]
_DISTRICTS_UNDER_PROVINCES_INFO = GOVS_INFO[GOVS_INFO["level"] == "省辖县"]

CITY_LEVEL_INFO = GOVS_INFO[GOVS_INFO["level"].isin(["市", "直辖市"])]
DISTRICT_LEVEL_INFO = GOVS_INFO[GOVS_INFO["level"].isin(["区县", "省辖县"])]


def get_gov_level_info(gov_level: GovLevel | int) -> pd.DataFrame:
    if isinstance(gov_level, int):
        gov_level = GovLevel(gov_level)

    if gov_level == GovLevel.ALL:
        return GOVS_INFO
    if gov_level == GovLevel.NATION:
        return NATION_INFO
    if gov_level == GovLevel.PROVINCE:
        return PROVINCES_INFO
    if gov_level == GovLevel.CITY_LEVEL:
        return CITY_LEVEL_INFO
    if gov_level == GovLevel.DISTRICT_LEVEL:
        return DISTRICT_LEVEL_INFO

    raise ValueError("Unknown gov_level")


def get_gov_level_ids(gov_level: GovLevel | int) -> pd.Index:
    return get_gov_level_info(gov_level).index


DAAS_INDEX_LIST = pd.read_excel("./utils/daas_index_list.xlsx", index_col=0)


def get_values_one_month(month: int, gov_ids: list, index_ids: list) -> pd.DataFrame:
    to_be_saved = pd.DataFrame(
        {"gov_id": pd.Series([], dtype="int"), "month": pd.Series([], dtype="int")}
    )

    for i, gov_id in enumerate(gov_ids):
        print(f"Progress: {i + 1} / {len(gov_ids)}")
        row = {
            "gov_id": gov_id,
            "month": month,
        }

        api_ret = zk2861api.api_get_gov_indexs_data(
            SECURE_KEY,
            gov_id,
            index_ids,
            f"{month_int_to_str(month)}-01",
            timeout=60,
        )
        data = api_ret["data"]
        for col in data:
            row[col["index_id"]] = col["value"]

        to_be_saved = to_be_saved.append(row, ignore_index=True)

    return to_be_saved


def calc_relative_positions(series: pd.Series) -> pd.Series:
    min_value = min(series)
    max_value = max(series)
    dist = max_value - min_value
    return (series - min_value) / dist


def sort_series_and_calc_ordinal_numbers(series: pd.Series) -> pd.Series:
    """
    Parameters:
        series: should not have NaN values.
    """
    sorted_series = series.sort_values(ascending=False).reset_index(name="value")
    drop_dup = sorted_series.drop_duplicates(subset=["value"])
    dropped = pd.concat([drop_dup, sorted_series]).drop_duplicates(keep=False)

    def reassign_index(row: pd.Series) -> pd.Series:
        row["value"] = drop_dup.loc[drop_dup["value"] == row["value"]].index[0]
        return row

    set_good_index = dropped.apply(reassign_index, axis=1)
    drop_dup["value"] = drop_dup.index
    together = pd.concat([drop_dup, set_good_index]).set_index("index")["value"]

    return together


class MeasureOfDispersion(enum.Enum):
    SD = "Standard deviation"
    IQR = "Interquartile range"
    CV = "Coefficient of variation"
    CV2 = "Absolute value of CV (ddof=1)"
    QCD = "Quartile coefficient of dispersion"


def calc_dispersions(
    input_df: pd.DataFrame, measure_of_dispersion: MeasureOfDispersion
) -> pd.Series:
    if measure_of_dispersion == MeasureOfDispersion.SD:
        dispersions = input_df.std()
    elif measure_of_dispersion == MeasureOfDispersion.IQR:
        dispersions = input_df.quantile(3 / 4) - input_df.quantile(1 / 4)
    elif measure_of_dispersion == MeasureOfDispersion.CV:
        dispersions = pd.Series(
            variation(input_df, nan_policy="omit"), index=input_df.columns
        )
    elif measure_of_dispersion == MeasureOfDispersion.CV2:
        dispersions = np.abs(
            pd.Series(
                variation(input_df, nan_policy="omit", ddof=1), index=input_df.columns
            )
        )
    elif measure_of_dispersion == MeasureOfDispersion.QCD:
        diff = input_df.quantile(3 / 4) - input_df.quantile(1 / 4)
        sums = input_df.quantile(3 / 4) + input_df.quantile(1 / 4)
        dispersions = diff / sums

    return dispersions


# Preset features


PRESET_FEATURES: dict[str | None, list] = {
    None: [],
    "[6-9]0后人口占常住人口的比重（年度）": [154518, 154520, 154522, 154524],
    "城镇居民人均[可支配收入,消费支出]": [125, 623],
    "[官媒,民众]水污染[治理,抱怨]网络热度指数": [26, 35],
}


class ValueOntology(enum.Enum):
    UNDETERMINED = 0
    QUANTITY = 1
    PERCENTAGE = 2
    DIFF_QUANTITY = 3
    DIFF_RATE = 4
    SCORE = 5
    QUANTITY_PER_CAPITA = 6
    PRICE = 7
    PROPORTION_OF_NATION = 8


VALUE_ONTOLOGY_CAN_BE_DIVIDED_BY_POPULATION = {
    ValueOntology.UNDETERMINED: False,
    ValueOntology.QUANTITY: True,
    ValueOntology.PERCENTAGE: False,
    ValueOntology.DIFF_QUANTITY: True,
    ValueOntology.DIFF_RATE: False,
    ValueOntology.SCORE: False,
    ValueOntology.QUANTITY_PER_CAPITA: False,
    ValueOntology.PRICE: False,
    ValueOntology.PROPORTION_OF_NATION: True,
}


def calc_value_ontology(name: str, unit: str) -> ValueOntology:
    quantity_positions = [
        name.rfind("数量"),
        name.rfind("个数"),
        name.rfind("数"),
        name.rfind("量"),
        name.rfind("次"),
        name.rfind("值"),
    ]

    percentage_positions = [
        name.rfind("占比"),
        name.rfind("比重"),
        name.rfind("比"),
        name.rfind("率"),
    ]

    diff_positions = [
        name.rfind("增长"),
        name.rfind("增长数量"),
        name.rfind("增"),
        name.rfind("增加值"),
        name.rfind("改善"),
        name.rfind("变化"),
        name.rfind("新增"),
    ]

    score_positions = [
        name.rfind("指数"),
        name.rfind("得分"),
        name.rfind("评分"),
        name.rfind("分"),
        name.rfind("级"),
        name.rfind("度"),
    ]

    quantity_per_capita_positions = [
        name.rfind("每万人"),
        name.rfind("人均"),
    ]

    price_positions = [
        name.rfind("价格"),
        name.rfind("资本"),
        name.rfind("价"),
    ]

    unit_is_percentage = unit in ["%", "‰"]

    # Anomalies
    if name == "每万人保险业企业数量":
        return ValueOntology.QUANTITY_PER_CAPITA
    if name == "每万人保险业个体工商户数量":
        return ValueOntology.QUANTITY_PER_CAPITA
    if name == "其他金融业企业平均注册资本":
        return ValueOntology.PRICE

    # pylint: disable=fixme
    # Todo: come up with even more classes
    if name == "近2年城镇居民人均可支配收入增长":
        return ValueOntology.QUANTITY_PER_CAPITA
    if name == "近2年城镇居民人均消费支出增长":
        return ValueOntology.QUANTITY_PER_CAPITA

    # Very specific
    if name == "空气质量同比改善率":
        return ValueOntology.DIFF_RATE
    if name == "空气质量优良天数比例":
        return ValueOntology.PERCENTAGE
    if "空气质量" in name:
        return ValueOntology.SCORE
    if name[-1] == "力" and unit == "万人次":
        return ValueOntology.QUANTITY
    if name[-1] == "度" and unit == "万人次":
        return ValueOntology.QUANTITY
    if name[-2:] == "速度":
        return ValueOntology.SCORE
    if name[-2:] == "面积":
        return ValueOntology.SCORE
    if "人口" in name and "增长" not in name and unit == "万人":
        return ValueOntology.QUANTITY
    if "增加值" in name and unit == "万元":
        return ValueOntology.DIFF_QUANTITY
    if "增长率" in name and unit_is_percentage:
        return ValueOntology.DIFF_RATE
    if name in ["城镇居民人均可支配收入", "城镇居民人均消费支出"]:
        return ValueOntology.QUANTITY_PER_CAPITA
    if "总价" in name and unit in ["元", "万元"]:
        return ValueOntology.QUANTITY
    if "全国占比" in name and unit_is_percentage:
        return ValueOntology.PROPORTION_OF_NATION
    if "增" in name and "数量" in name and not unit_is_percentage:
        return ValueOntology.DIFF_QUANTITY

    # More specific
    unit_is_score = "分" in unit or "级" in unit
    unit_is_per_capita = "/人" in unit or "/万人" in unit
    unit_is_price = "元" in unit

    if max(score_positions) != -1 and unit_is_score:
        return ValueOntology.SCORE
    if max(quantity_per_capita_positions) != -1 and unit_is_per_capita:
        return ValueOntology.QUANTITY_PER_CAPITA
    if max(price_positions) != -1 and unit_is_price:
        return ValueOntology.PRICE

    # More general
    three_types_positions = quantity_positions + percentage_positions + diff_positions
    three_types_positions.sort()
    last = three_types_positions[-1]

    if last == -1:
        return ValueOntology.UNDETERMINED
    if last in quantity_positions and (not unit_is_percentage):
        return ValueOntology.QUANTITY
    if last in percentage_positions and unit_is_percentage:
        return ValueOntology.PERCENTAGE
    if last in diff_positions and (not unit_is_percentage):
        return ValueOntology.DIFF_QUANTITY
    if last in diff_positions and unit_is_percentage:
        return ValueOntology.DIFF_RATE

    return ValueOntology.UNDETERMINED


_MACRO_DATA_NODES_LIST = pd.read_excel(
    "./utils/macro_data_nodes_list.xlsx", index_col=0
)
MACRO_DATA_DIR_INDEX_NODES = _MACRO_DATA_NODES_LIST.loc[
    _MACRO_DATA_NODES_LIST["node_type"] == "dir_index"
].set_index("node_id", drop=True)


# Gov classification


# 常住人口（月度）（运营商信令算法T-APP）
POP_MONTH_INDEX = 65583


class DistrictsPopulationLevel(enum.Enum):
    LOW = 1
    MID_LOW = 2
    MID_HIGH = 3
    HIGH = 4


DISTRICTS_POPULATION_LEVELS_RANGES: dict[DistrictsPopulationLevel, list] = {
    DistrictsPopulationLevel.LOW: [-np.inf, 50],
    DistrictsPopulationLevel.MID_LOW: [50, 150],
    DistrictsPopulationLevel.MID_HIGH: [150, 250],
    DistrictsPopulationLevel.HIGH: [250, np.inf],
}


def calc_district_population_level(population: float) -> DistrictsPopulationLevel:
    for pop_level in DistrictsPopulationLevel:
        lower, upper = DISTRICTS_POPULATION_LEVELS_RANGES[pop_level]
        if lower <= population < upper:
            return pop_level
    raise ValueError("Cannot determine district population level.")


# 城镇居民人均可支配收入
INCOME_PER_CAPITA_INDEX = 125


class DistrictsIncomeLevel(enum.Enum):
    LOW = 1
    MID_LOW = 2
    MID_HIGH = 3
    HIGH = 4


DISTRICTS_INCOME_LEVELS_RANGES: dict[DistrictsIncomeLevel, list] = {
    DistrictsIncomeLevel.LOW: [-np.inf, 3],
    DistrictsIncomeLevel.MID_LOW: [3, 4],
    DistrictsIncomeLevel.MID_HIGH: [4, 6.5],
    DistrictsIncomeLevel.HIGH: [6.5, np.inf],
}


def calc_district_income_level(income: float) -> DistrictsIncomeLevel:
    for income_level in DistrictsIncomeLevel:
        lower, upper = DISTRICTS_INCOME_LEVELS_RANGES[income_level]
        if lower <= income < upper:
            return income_level
    raise ValueError("Cannot determine district income level.")


def calc_seven_areas_map_area_to_gov_ids() -> dict[str, list[int]]:
    def get_all_gov_ids(gov_dict: dict) -> list[int]:
        this_id_list = [gov_dict["gov_id"]] if "gov_id" in gov_dict else []
        all_descendants_ids = list(
            chain.from_iterable(
                [get_all_gov_ids(child_dict) for child_dict in gov_dict["children"]]
            )
        )
        return [*this_id_list, *all_descendants_ids]

    return {
        area_dict["name"]: get_all_gov_ids(area_dict)  # type: ignore
        for area_dict in gov_bar_data.user_govs_region
    }


SEVEN_AREAS_MAP_AREA_TO_GOV_IDS = calc_seven_areas_map_area_to_gov_ids()


def calc_district_seven_areas_level(gov_id: int) -> str:
    # Missing
    if gov_id in [3235, 3236, 3238, 3239, 3240, 3241, 3242]:
        return "东北地区"
    # Standard
    for area_name in SEVEN_AREAS_MAP_AREA_TO_GOV_IDS:
        if gov_id in SEVEN_AREAS_MAP_AREA_TO_GOV_IDS[area_name]:
            return area_name
    raise ValueError("Cannot determine seven areas level.")


EIGHTY_TWO_INDUSTRIES = pd.read_excel("./utils/industry_index_info.xlsx")


# 文字解读


def correlation_interpretation(
    index_id: int,
    index_name: str,
    other_index_id: int,
    other_index_name: str,
    value: float,
) -> str:
    zheng_or_fu = "正" if value > 0 else "负"
    number = round(value, 2)
    return f"“{index_name}”与“{other_index_name}”具有强{zheng_or_fu}相关性，相关度为{number}"


def frequent_item_sets_interpretation(item_sets: list[str], support: float) -> str:
    itemsets_better = "、".join([f"“{item}”" for item in item_sets])
    percent = round(support * 100, 2)
    return f"同时满足{itemsets_better}的区县众多，达到该类区县的{percent}%"


def sample_selection_districts_interpretation(
    gov_id: int, gov_name: str, districts_count: int
) -> str:
    return f"{gov_name}为全国{districts_count}个区县之一"


def sample_selection_districts_population_interpretation(
    gov_id: int, gov_name: str, count: int, district_pop_level: DistrictsPopulationLevel
) -> str:
    lower, upper = DISTRICTS_POPULATION_LEVELS_RANGES[district_pop_level]
    if lower == -np.inf:
        pop_itp = f"人口小于{upper}万"
    elif upper == np.inf:
        pop_itp = f"人口超过{lower}万"
    else:
        pop_itp = f"人口处于{lower}到{upper}万之间"
    return f"{gov_name}为{pop_itp}的区县，全国共有{count}个这样的区县"


def sample_selection_districts_income_interpretation(
    gov_id: int, gov_name: str, count: int, district_income_level: DistrictsIncomeLevel
) -> str:
    lower, upper = DISTRICTS_INCOME_LEVELS_RANGES[district_income_level]
    if lower == -np.inf:
        income_itp = f"人均可支配收入小于{upper}万"
    elif upper == np.inf:
        income_itp = f"人均可支配收入超过{lower}万"
    else:
        income_itp = f"人均可支配收入处于{lower}到{upper}万之间"
    return f"{gov_name}为{income_itp}的区县，全国共有{count}个这样的区县"


def sample_selection_districts_seven_areas_interpretation(
    gov_id: int, gov_name: str, count: int, district_seven_areas_level: Any
) -> str:
    return f"{gov_name}为{district_seven_areas_level.name}的区县，全国共有{count}个这样的区县"


def sample_selection_districts_provincial_capitals_children_interpretation(
    gov_id: int, gov_name: str, count: int
) -> str:
    return f"{gov_name}为省会下辖区县，全国共有{count}个这样的区县"


def sample_selection_districts_by_industry_interpretation(
    gov_id: int,
    gov_name: str,
    count: int,
    industry: Any,
) -> str:
    industry_name = industry.name
    industry_better = f"{industry_name[:industry_name.find('数量')]}富集度"
    return f"{gov_name}为{industry_better}最高的{count}个区县之一"


def sample_selection_time_series_interpretation(
    gov_id: int,
    gov_name: str,
    count: int,
    months: Any,
) -> str:
    return f"通过分析{gov_name}在{months.name}上的数据变化"
