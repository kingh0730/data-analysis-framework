# %%
from itertools import chain
from utils import gov_bar_data


def get_all_gov_ids(gov_dict: dict) -> list[int]:
    this_id_list = [gov_dict["gov_id"]] if "gov_id" in gov_dict else []
    all_descendants_ids = list(
        chain.from_iterable(
            [get_all_gov_ids(child_dict) for child_dict in gov_dict["children"]]
        )
    )
    return [*this_id_list, *all_descendants_ids]


def main() -> dict[str, list[int]]:
    return {
        area_dict["name"]: get_all_gov_ids(area_dict)
        for area_dict in gov_bar_data.user_govs_region
    }


if __name__ == "__main__":
    print(main())


# %%
