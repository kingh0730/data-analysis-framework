import my_package


def test_version() -> None:
    assert my_package.__version__ == "0.1.0"


def test_calc_value_ontology() -> None:
    test = my_package.calc_value_ontology("餐饮服务；外国餐厅；此类数据总个数", "个")
    assert test == my_package.ValueOntology.QUANTITY

    test = my_package.calc_value_ontology("空气质量优良天数比例", "%")
    assert test == my_package.ValueOntology.PERCENTAGE

    test = my_package.calc_value_ontology("近2年本地注册企业数量增长", "个")
    assert test == my_package.ValueOntology.DIFF_QUANTITY
