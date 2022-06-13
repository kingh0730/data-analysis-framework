# %%
import pandas as pd
import my_package as mp


# %%
pairs = pd.read_csv(
    "../data/2021-08_district-level_0.5_2_1418.5(Fri_Jun_10_20-05-34_2022).csv",
    index_col=0,
)
index1_and_index2 = pairs[["index1", "index2"]]
print(index1_and_index2)


# %%
indexes_info = pd.read_csv("../temp/daas_index_list_value_ontologies.csv", index_col=0)
value_ontologies = indexes_info["value_ontology"]
print(value_ontologies)


# %%
def calc_value_ontology_wrapper(
    index1_and_index2_pair: pd.Series,
) -> tuple[mp.ValueOntology, mp.ValueOntology]:
    ontology1 = value_ontologies[index1_and_index2_pair["index1"]]
    ontology2 = value_ontologies[index1_and_index2_pair["index2"]]
    return ontology1, ontology2


value_ontologies_pairs = index1_and_index2.apply(calc_value_ontology_wrapper, axis=1)
print(value_ontologies_pairs)


# %%
pairs[["value_ontology1", "value_ontology2"]] = pd.DataFrame(
    value_ontologies_pairs.tolist(), index=value_ontologies_pairs.index
)
pairs.to_csv("../temp/2021-08_district-level.csv")
print(pairs)


# %%
