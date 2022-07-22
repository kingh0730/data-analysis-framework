# coding:utf-8
# Created:  2022-01-07
# @author:  ke.li
# descript: daas interface

import os
import sys
import json
import random
import requests
import platform
import base64
try:
    # on linux
    from Crypto.Cipher import DES
except Exception:
    # on windows
    from Cryptodome.Cipher import DES


API_DOMAIN = 'https://api.zhiku2861.com'
ENCRYPT_ENABLE = 1

RANDOM_INT_START = 1000
RANDOM_INT_END = 9000

CODE_SUCCESS = 200
CODE_NET_ERR = 507
CODE_UNKNOW_ERR = 599


def post_request(url, param, timeout):

    if ENCRYPT_ENABLE == 1:
        transmit_param = my_encrypt(json.dumps(param, ensure_ascii=False))
    else:
        transmit_param = json.dumps(param, ensure_ascii=False)

    try:
        request_ret = requests.post(url, {'param': transmit_param, 'encrypt': ENCRYPT_ENABLE}, timeout=timeout)
        if request_ret.status_code == 200:
            request_resp = json.loads(request_ret.text)
            if request_resp['code'] == CODE_SUCCESS:
                if ENCRYPT_ENABLE == 1:
                    try:
                        request_resp['data'] = json.loads(my_decrypt(request_resp['data']))
                    except Exception as e:
                        return {'code': False, 'result': 'decrypt response crash: %s' % str(e)}
            return request_resp
        else:
            return {'code': CODE_NET_ERR, 'result': 'post %s fail: status_code = %s' % (url, request_ret.status_code)}
    except Exception as e:
        return {'code': CODE_NET_ERR, 'result': 'post %s crash = %s' % (url, str(e))}


def my_encrypt(text, fill_bit=8):

    random_key = str(random.randint(RANDOM_INT_START, RANDOM_INT_END))
    my_key = '{zk%s}' % random_key
    utf8_text = text.encode('utf8')
    temp_obj = DES.new(my_key.encode('utf8'), DES.MODE_ECB)
    encrypt_text = base64.standard_b64encode(temp_obj.encrypt(utf8_text + (fill_bit - (len(utf8_text) % fill_bit)) * " ".encode('utf8'))).decode("utf8")
    return 'zk' + encrypt_text + random_key


def my_decrypt(text):

    my_key = '{zk%s}' % text[-4:]
    temp_obj = DES.new(my_key.encode('utf8'), DES.MODE_ECB)
    return temp_obj.decrypt(base64.standard_b64decode(text[2:-4])).decode().rstrip()


# 获取指标列
def api_get_index_list(secure_key, timeout):

    request_url = '%s/info/index_list/v1' % API_DOMAIN
    auth_param = {'secure_key': secure_key, 'cpu_id': G_CPU_ID}
    func_param = {}

    return post_request(request_url, {'auth_param': auth_param, 'func_param': func_param}, timeout)


# 获取地区列表
def api_get_region_list(secure_key, gov_id, timeout):

    request_url = '%s/info/region_list/v1' % API_DOMAIN
    auth_param = {'secure_key': secure_key, 'cpu_id': G_CPU_ID}
    func_param = {'gov_id': gov_id}

    return post_request(request_url, {'auth_param': auth_param, 'func_param': func_param}, timeout)


# 获取网格列表
def api_get_grid_list(secure_key, gov_id, timeout):

    request_url = '%s/info/grid_list/v1' % API_DOMAIN
    auth_param = {'secure_key': secure_key, 'cpu_id': G_CPU_ID, }
    func_param = {'gov_id': gov_id}

    return post_request(request_url, {'auth_param': auth_param, 'func_param': func_param}, timeout)


# 获取当前账户已授权了的指标列表
def api_get_permit_index_list(secure_key, timeout):

    request_url = '%s/permit/index_list/v1' % API_DOMAIN
    auth_param = {'secure_key': secure_key, 'cpu_id': G_CPU_ID}
    func_param = {}

    return post_request(request_url, {'auth_param': auth_param, 'func_param': func_param}, timeout)


# 获取当前账户已授权了的区县GIS库地区列表
def api_get_permit_gov_region_list(secure_key, timeout):

    request_url = '%s/permit/gov_region_list/v1' % API_DOMAIN
    auth_param = {'secure_key': secure_key, 'cpu_id': G_CPU_ID}
    func_param = {}

    return post_request(request_url, {'auth_param': auth_param, 'func_param': func_param}, timeout)


# 获取当前账户已授权了的网格GIS库地区列表
def api_get_permit_grid_region_list(secure_key, timeout):

    request_url = '%s/permit/grid_region_list/v1' % API_DOMAIN
    auth_param = {'secure_key': secure_key, 'cpu_id': G_CPU_ID}
    func_param = {}

    return post_request(request_url, {'auth_param': auth_param, 'func_param': func_param}, timeout)


# 获取当前账户已授权了的选点GIS库地区列表
def api_get_permit_point_region_list(secure_key, timeout):

    request_url = '%s/permit/point_region_list/v1' % API_DOMAIN
    auth_param = {'secure_key': secure_key, 'cpu_id': G_CPU_ID}
    func_param = {}

    return post_request(request_url, {'auth_param': auth_param, 'func_param': func_param}, timeout)


# 获取某个省市区县地区的多个指标数据
def api_get_gov_indexs_data(secure_key, gov_id, index_ids, date, timeout):

    request_url = '%s/gov_data/gov_indexs/v1' % API_DOMAIN
    auth_param = {'secure_key': secure_key, 'cpu_id': G_CPU_ID, }
    func_param = {'gov_id': gov_id, 'index_ids': index_ids, 'date': date}

    return post_request(request_url, {'auth_param': auth_param, 'func_param': func_param}, timeout)


# 获取多个省市区县地区的某个指标数据
def api_get_govs_index_data(secure_key, gov_ids, index_id, date, timeout):

    request_url = '%s/gov_data/govs_index/v1' % API_DOMAIN
    auth_param = {'secure_key': secure_key, 'cpu_id': G_CPU_ID, }
    func_param = {'gov_ids': gov_ids, 'index_id': index_id, 'date': date}

    return post_request(request_url, {'auth_param': auth_param, 'func_param': func_param}, timeout)


# 获取某个网格的多个指标数据
def api_get_grid_indexs_data(secure_key, grid_id, location, index_ids, date, timeout):

    request_url = '%s/grid_data/grid_indexs/v1' % API_DOMAIN
    auth_param = {'secure_key': secure_key, 'cpu_id': G_CPU_ID, }
    func_param = {'grid_id': grid_id, 'location': location, 'index_ids': index_ids, 'date': date}

    return post_request(request_url, {'auth_param': auth_param, 'func_param': func_param}, timeout)


# 获取一个城市或区县所有网格的某个指标数据
def api_get_grids_index_data(secure_key, gov_id, index_id, date, timeout):

    request_url = '%s/grid_data/grids_index/v1' % API_DOMAIN
    auth_param = {'secure_key': secure_key, 'cpu_id': G_CPU_ID, }
    func_param = {'gov_id': gov_id, 'index_id': index_id, 'date': date}

    return post_request(request_url, {'auth_param': auth_param, 'func_param': func_param}, timeout)


# 调用账号自定义接口
def api_customize_interface(secure_key, interface_id, interface_param, timeout):

    request_url = '%s/customize/interface/v1' % API_DOMAIN
    auth_param = {'secure_key': secure_key, 'cpu_id': G_CPU_ID, }
    func_param = {'interface_id': interface_id, 'interface_param': json.dumps(interface_param)}

    return post_request(request_url, {'auth_param': auth_param, 'func_param': func_param}, timeout)


def cpu_id():

    if platform.system() == 'Windows':
        cpu_info = wmi.WMI()
        for cpu in cpu_info.Win32_Processor():
            return cpu.ProcessorId.strip().upper()
    else:
        cmd = 'dmidecode -t 4 | grep ID'
        r = os.popen(cmd)
        text = r.read().strip().upper()
        r.close()
        text = text.replace(' ', '')
        return text.replace('ID:', '')

    return None


if platform.system() == 'Windows':
    import wmi

G_CPU_ID = cpu_id()
