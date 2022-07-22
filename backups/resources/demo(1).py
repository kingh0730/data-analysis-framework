# coding:utf-8
# Created:  2017-02-17
# @author:  ke.li
# descript: result class

import lib.io.zk2861api as zk2861api

SECURE_KEY = 'NDRkOGJlZTFkMDQzMjUwZTQ0ZjIyMmU5YjIwY2U0NDU='
zk2861api.ENCRYPT_ENABLE = 0
DEMO_GET_DATA_DATE = '2021-04-30'


def demo_api_get_index_list():

    api_ret = zk2861api.api_get_index_list(SECURE_KEY, timeout=30)
    pass


def demo_api_get_region_list():

    api_ret = zk2861api.api_get_region_list(SECURE_KEY, 0, timeout=30)
    pass


def demo_api_get_grid_list():

    api_ret = zk2861api.api_get_grid_list(SECURE_KEY, 1, timeout=600)
    pass


def demo_api_get_permit_index_list():

    api_ret = zk2861api.api_get_permit_index_list(SECURE_KEY, timeout=30)
    pass


def demo_api_get_permit_gov_region_list():

    api_ret = zk2861api.api_get_permit_gov_region_list(SECURE_KEY, timeout=30)
    pass


def demo_api_get_permit_grid_region_list():

    api_ret = zk2861api.api_get_permit_grid_region_list(SECURE_KEY, timeout=30)
    pass


def demo_api_get_permit_point_region_list():

    api_ret = zk2861api.api_get_permit_point_region_list(
        SECURE_KEY, timeout=30)
    pass


def demo_api_get_gov_indexs_data():

    api_ret = zk2861api.api_get_gov_indexs_data(
        SECURE_KEY, 2, [620], DEMO_GET_DATA_DATE, timeout=60)
    pass


def demo_api_get_govs_index_data():

    api_ret = zk2861api.api_get_govs_index_data(
        SECURE_KEY, None, 620, DEMO_GET_DATA_DATE, timeout=60)
    pass


def demo_api_get_grid_indexs_data():

    api_ret = zk2861api.api_get_grid_indexs_data(
        SECURE_KEY, None, '106.674451,26.619442',  None, DEMO_GET_DATA_DATE, timeout=60)
    pass


def demo_api_get_grids_index_data():

    api_ret = zk2861api.api_get_grids_index_data(
        SECURE_KEY, 7, 188, DEMO_GET_DATA_DATE, timeout=60)
    pass


def demo_api_customize_interface():

    api_ret = zk2861api.api_customize_interface(
        SECURE_KEY, 1, dict(), timeout=60)
    pass


def demo_api_get_yearbook_version_data():

    api_ret = zk2861api.api_yearbook_version_data(
        SECURE_KEY, [1], ['2018-01-01', '2019-01-01'], timeout=60)
    pass


if __name__ == "__main__":

    pass

    # demo_api_get_index_list()
    # demo_api_get_region_list()
    # demo_api_get_grid_list()
    demo_api_get_permit_index_list()
    demo_api_get_permit_gov_region_list()
    # demo_api_get_permit_grid_region_list()
    # demo_api_get_permit_point_region_list()
    demo_api_get_gov_indexs_data()
    demo_api_get_govs_index_data()
    # demo_api_get_grid_indexs_data()
    # demo_api_get_grids_index_data()
    # demo_api_customize_interface()
    # demo_api_get_yearbook_version_data()
