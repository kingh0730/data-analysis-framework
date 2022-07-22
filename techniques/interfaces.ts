/**
 * 样本选择方法
 */
enum SampleSelectionTechnique {
  /**
   * 选择全部区县
   */
  Districts = "DISTRICTS",
  /**
   * 按照输入的gov_id选择所有在人口上与之匹配的区县
   */
  DistrictsPopulation = "DISTRICTS_POPULATION",
  /**
   * 按照输入的gov_id选择所有在人均可支配收入上与之匹配的区县
   */
  DistrictsIncome = "DISTRICTS_INCOME",
  /**
   * 按照输入的gov_id选择所有在地域上与之匹配的区县
   */
  DistrictsSevenAreas = "DISTRICTS_SEVEN_AREAS",
  /**
   * 选择所有省会下辖的区县
   */
  DistrictsProvincialCapitalsChildren = "DISTRICTS_PROVINCIAL_CAPITALS_CHILDREN",
  /**
   * 选择某行业富集度最高的N个区县
   */
  DistrictsByIndustry = "DISTRICTS_BY_INDUSTRY",
  /**
   * 选择gov_id过去N个月的全部数据
   */
  TimeSeries = "TIME_SERIES",
}

/**
 * 关联性挖掘方法
 */
enum RelationMiningTechnique {
  /**
   * 皮尔逊相关系数
   */
  PearsonCorr = "PEARSON_CORR",
  /**
   * 排名相关系数
   */
  SpearmanCorr = "SPEARMAN_CORR",
  /**
   * 频繁项集
   */
  FrequentItemSets = "FREQUENT_ITEM_SETS",
  /**
   * 关联规则
   */
  AssociationRules = "ASSOCIATION_RULES",
  /**
   * 时间序列
   */
  TimeSeries = "TIME_SERIES",
}

/**
 * 指标组筛选方法
 */
enum FilterTechnique {
  /**
   * 去除所有指标都具有相同类型的指标组
   */
  NoSameType = "NO_SAME_TYPE",
  /**
   * 去除所有指标都具有相同子类型的指标组
   */
  NoSameSubType = "NO_SAME_SUB_TYPE",
  /**
   * 去除所有以年为单位更新的指标
   */
  NoYearPeriod = "NO_YEAR_PERIOD",
  /**
   * 去除所有离散度小于x的指标
   */
  MinDispersion = "MIN_DISPERSION",
}

/**
 * 输入接口
 */
interface Input {
  /**
   * 寻找此index_id的关联性
   */
  index_id: number;
  /**
   * 根据此gov_id寻找关联性
   */
  gov_id: number;
  /**
   * 根据此时间（版本）寻找关联性
   */
  month: string;
  /**
   * 样本选择方法的设置
   */
  sample_selection: {
    /**
     * 一个样本选择方法
     */
    technique: SampleSelectionTechnique;
    /**
     * 该样本选择所需要的参数
     */
    kwargs: { [key: string]: any };
  };
  /**
   * 关联性挖掘方法的设置
   */
  relation_mining: {
    /**
     * 一个关联性挖掘方法
     */
    technique: RelationMiningTechnique;
    /**
     * 该关联性挖掘方法所需要的参数
     */
    kwargs: { [key: string]: any };
  };
  /**
   * 指标组筛选方法的设置（多个方法）
   */
  filter: {
    /**
     * 一个指标组筛选方法
     */
    technique: FilterTechnique;
    /**
     * 该指标组筛选方法所需要的参数
     */
    kwargs: { [key: string]: any };
  }[];
}

/**
 * 输出接口
 */
interface Output {
  /**
   * 样本选择的结果
   */
  samples: {
    /**
     * 所有样本选择的条件（多个条件）
     */
    criteria: {
      /**
       * 样本选择条件的名称
       */
      name: string;
      /**
       * 样本选择条件下的类别
       */
      class: string;
    }[];
    /**
     * 共有多少样本
     */
    count: number;
    /**
     * 样本选择结果的解读
     */
    interpretation: string;
  };
  /**
   * 挖掘出的所有关联性
   */
  relations: {
    /**
     * 与输入的index_id关联的另一个index_id
     */
    index_id: number;
    /**
     * 关联性挖掘方法的名称
     */
    technique: string;
    /**
     * 所有在该关联性挖掘方法下产生的关联性的值
     */
    values: {
      /**
       * 关联性值的名称
       */
      name: string;
      /**
       * 关联性的值
       */
      value: number;
      /**
       * 关联性值的解读
       */
      interpretation: string;
    }[];
    /**
     * 在该关联性挖掘方法下产生的有可能有价值的副产品
     */
    byproducts: { [key: string]: any };
    /**
     * 与输入的gov_id有关系的有可能有价值的副产品
     */
    gov_specific: { [key: string]: any };
  }[];
  /**
   * 综合样本选择和关联性挖掘的所有结果的解读
   */
  interpretation: string;
}
