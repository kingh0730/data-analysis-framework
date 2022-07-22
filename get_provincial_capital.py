# %%
from itertools import chain
from utils.gov_bar_data import user_govs_region
from utils import PROVINCIAL_CAPITALS_NAMES


# %%
user_govs_region


# %%
PROVINCIAL_CAPITALS_NAMES


# %%
len(PROVINCIAL_CAPITALS_NAMES)


# %%
def get_all_govs_and_children() -> list[dict]:
    def get_all_govs_wrapper(gov_dict: dict) -> list[dict]:
        this_id_list = [gov_dict] if "gov_id" in gov_dict else []
        all_descendants_ids = list(
            chain.from_iterable(
                [
                    get_all_govs_wrapper(child_dict)
                    for child_dict in gov_dict["children"]
                ]
            )
        )
        return [*this_id_list, *all_descendants_ids]

    return list(
        chain.from_iterable(
            [get_all_govs_wrapper(area_dict) for area_dict in user_govs_region]
        )
    )


# %%
all_govs_and_children = get_all_govs_and_children()
all_govs_and_children


# %%
all_provincial_capitals = [
    gov
    for gov in all_govs_and_children
    if gov["name"] in [f"{name}市" for name in PROVINCIAL_CAPITALS_NAMES]
]
all_provincial_capitals


# %%
len(all_provincial_capitals)


# %%
all_provincial_capitals_children = [
    child | {"parent_id": capital["gov_id"], "parent_name": capital["name"]}
    for capital in all_provincial_capitals
    for child in capital["children"]
]
all_provincial_capitals_children


# %%
[child for child in all_provincial_capitals_children if child["children"] != []]
