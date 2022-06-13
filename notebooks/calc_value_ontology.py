# %%
import pandas as pd
import my_package as mp


# %%
indexes_info = pd.read_excel("../data/daas_index_list.xlsx", index_col=0)
names_and_units = indexes_info[["index_name", "unit"]]
print(names_and_units)


# %%
def calc_value_ontology_wrapper(name_and_unit: pd.Series) -> mp.ValueOntology:
    ontology = mp.calc_value_ontology(
        name_and_unit["index_name"], name_and_unit["unit"]
    )
    return ontology


value_ontologies = names_and_units.apply(calc_value_ontology_wrapper, axis=1)
print(value_ontologies)


# %%
indexes_info["value_ontology"] = value_ontologies
indexes_info.to_csv("../temp/daas_index_list_value_ontologies.csv")
print("Done.")


# %%
