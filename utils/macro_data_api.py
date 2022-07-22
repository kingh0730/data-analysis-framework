# 获取宏观指标的HTTP 接口


# API: "https://cy_test.daasmart.com/qry_cy_app_data/api4data/api_gov_risk/gov_more_macro_data"

#    gov_ids: 按 "," 分隔的gov_id序列

# 参数：         {"gov_ids": gov_ids,
#                 "node_id": node_id,
#                 "version": version,
#                 "version_cnt": 0,
#                 "value_type": "real",
#                 "api_name": "api_govs_vers_node",
#                 "user_id": "uqtyBhWEy4 Ds5nr8g5nFQ==",
#                 "refer_product": "unicom",
#                 }


import json
import asyncio
from typing import Any, Iterable
import aiohttp
import pandas as pd

import requests


API_DOMAIN = "https://cy_test.daasmart.com/qry_cy_app_data/api4data/api_gov_risk/gov_more_macro_data"


CODE_NET_ERR = 507


async def gather_with_concurrency(number_of_concurrency: int, *tasks: Any) -> Any:
    semaphore = asyncio.Semaphore(number_of_concurrency)

    async def sem_task(task: Any) -> Any:
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks))


# aiohttp


async def fetch_get(session: aiohttp.ClientSession, url: str) -> Any:
    async with session.get(url) as response:
        return await response.text()


async def fetch_post(session: aiohttp.ClientSession, url: str, param: Any) -> Any:
    # print(param["node_id"])
    async with session.post(url, data=param) as response:
        return await response.text()


def api_get_macro_data_gov_nodes(
    gov_id: int, node_ids: list[str], time_str: str
) -> pd.DataFrame:
    async def helper() -> Any:
        params = [
            {
                "gov_ids": gov_id,
                "node_id": node_id,
                "version": time_str,
                "version_cnt": 0,
                "value_type": "real",
                "api_name": "api_govs_vers_node",
                "user_id": "uqtyBhWEy4 Ds5nr8g5nFQ==",
                "refer_product": "unicom",
            }
            for node_id in node_ids
        ]
        tasks = []
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(verify_ssl=False)
        ) as session:
            for param in params:
                tasks.append(fetch_post(session, API_DOMAIN, param))
            res_results = await gather_with_concurrency(10, *tasks)

            res_dfs = []
            for res_str in res_results:
                res = json.loads(res_str)
                res_df = pd.DataFrame(
                    {
                        k: l
                        for k, l in res["data"].items()
                        if k
                        in ["type_code", "data_type", "version", "gov_id", "node_sum"]
                    }
                )
                res_dfs.append(res_df)
            return pd.concat(res_dfs).sort_values("type_code", ascending=True)

    return asyncio.run(helper())


# Incorrect


def _post_request(url: str, param: Any, timeout: int) -> Any:
    try:
        request_ret = requests.post(url, param, timeout=timeout)
        if request_ret.status_code == 200:
            request_resp = json.loads(request_ret.text)
            return request_resp
        else:
            return {
                "code": CODE_NET_ERR,
                "result": f"post {url} fail: status_code = {request_ret.status_code}",
            }
    except requests.ConnectionError as exception:
        return {
            "code": CODE_NET_ERR,
            "result": f"post {url} crash = {exception}",
        }


def _api_get_macro_data(gov_ids: str, node_id: str, version: str, timeout: int) -> Any:
    request_url = API_DOMAIN
    param = {
        "gov_ids": gov_ids,
        "node_id": node_id,
        "version": version,
        "version_cnt": 0,
        "value_type": "real",
        "api_name": "api_govs_vers_node",
        "user_id": "uqtyBhWEy4 Ds5nr8g5nFQ==",
        "refer_product": "unicom",
    }
    return _post_request(request_url, param, timeout)


async def _api_get_macro_data_govs_node(
    gov_ids: Iterable[int], node_id: str, time_str: str
) -> pd.DataFrame:
    print(gov_ids, node_id, time_str)
    res = _api_get_macro_data(
        ",".join(map(str, gov_ids)), node_id, time_str, timeout=60
    )
    res_df = pd.DataFrame(
        {
            k: l
            for k, l in res["data"].items()
            if k in ["type_code", "data_type", "version", "gov_id", "node_sum"]
        }
    )
    return res_df


def sync_api_get_macro_data_gov_nodes(
    gov_id: int, node_ids: list[str], time_str: str
) -> pd.DataFrame:
    async def helper() -> Any:
        return await asyncio.gather(
            *[
                _api_get_macro_data_govs_node([gov_id], node_id, time_str)
                for node_id in node_ids
            ]
        )

    tasks_done = asyncio.run(helper())
    res = pd.concat(tasks_done).sort_values("type_code", ascending=True)

    return res
