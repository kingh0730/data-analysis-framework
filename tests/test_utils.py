import utils


def test_version() -> None:
    assert utils.__version__ == "0.1.0"


def test_calc_value_ontology() -> None:
    test = utils.calc_value_ontology("餐饮服务；外国餐厅；此类数据总个数", "个")
    assert test == utils.ValueOntology.QUANTITY

    test = utils.calc_value_ontology("空气质量优良天数比例", "%")
    assert test == utils.ValueOntology.PERCENTAGE

    test = utils.calc_value_ontology("近2年本地注册企业数量增长", "个")
    assert test == utils.ValueOntology.DIFF_QUANTITY
