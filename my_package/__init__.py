__version__ = "0.1.0"


import enum


class ValueOntology(enum.Enum):
    UNDETERMINED = 0
    QUANTITY = 1
    PERCENTAGE = 2
    DIFF_QUANTITY = 3
    DIFF_PERCENTAGE = 4


def calc_value_ontology(name: str, unit: str) -> ValueOntology:
    quantity_positions = [
        name.rfind("数量"),
        name.rfind("个数"),
        name.rfind("数"),
        name.rfind("量"),
        name.rfind("次"),
        name.rfind("分"),
        name.rfind("级"),
    ]

    percentage_positions = [
        name.rfind("占比"),
        name.rfind("比重"),
        name.rfind("比"),
    ]

    diff_positions = [
        name.rfind("增长"),
        name.rfind("增"),
    ]

    all_positions = quantity_positions + percentage_positions + diff_positions
    all_positions.sort()
    last = all_positions[-1]

    unit_is_percentage = unit in ["%", "‰"]

    if last == -1:
        return ValueOntology.UNDETERMINED
    if last in quantity_positions and (not unit_is_percentage):
        return ValueOntology.QUANTITY
    if last in percentage_positions and unit_is_percentage:
        return ValueOntology.PERCENTAGE
    if last in diff_positions and (not unit_is_percentage):
        return ValueOntology.DIFF_QUANTITY
    if last in diff_positions and unit_is_percentage:
        return ValueOntology.DIFF_PERCENTAGE

    return ValueOntology.UNDETERMINED
