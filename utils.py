__author__ = 'yihanjiang'
import torch
import numpy as np
import math
from typing import List
from itertools import product
import re

def replace_and_round(arr: np.ndarray, x: float) -> List[List[int]]:
    """输入为二维ndarray，输出为将所有[0.5-x, 0.5+x]区间内的数替换为0或1，在区间范围外的就四舍五入成0或1的所有可能的ndarray"""
    mask = (arr >= (0.5 - x)) & (arr <= (0.5 + x))
    indices = np.where(mask)
    # 生成所有可能的替换组合 (0 和 1 的组合)
    combinations = list(product([0, 1], repeat=len(indices[0])))
    result = []
    for combo in combinations:
        new_arr = arr.copy()
        new_arr[indices] = combo
        new_arr[~mask] = np.round(new_arr[~mask])
        result.append(new_arr.astype(int))
    return result


def to_asc(strings: List[str]) -> List[List[int]]:
    """将字符串列表转换为ASCII码的二进制表示列表"""
    asc_list = []
    for s in strings:
        bin_list = []
        for char in s:
            bin_str = format(ord(char), '08b')
            bin_list.extend([int(bit) for bit in bin_str])
        asc_list.append(bin_list)
    return asc_list

def to_en(asc_list: List[List[int]]) -> List[str]:
    """将ASCII码的二进制表示列表转换回字符串列表"""
    strings = []
    for bin_list in asc_list:
        s = ""
        for i in range(0, len(bin_list), 8):
            byte = bin_list[i:i+8]
            char = chr(int(''.join(map(str, byte)), 2))
            s += char
        strings.append(s)
    return strings

def str_completion(args_len: int, sentences: List[List[int]]) -> List[List[int]]:
    """将长度不足的句子补全为args_len的长度"""
    for i in range(len(sentences)):
        if len(sentences[i]) < args_len:
            sentences[i] += "*" * (args_len - len(sentences[i]))

    return sentences


def is_valid_string(s: str) -> bool:
    """使用正则表达式来匹配只包含大小写英文字母和常见英文标点符号的字符串"""
    pattern = r'^[A-Za-z0-9 .,!?"\'():;_\-\[\]{}<>@#$%^&*+=|\\/~`]*$'
    return bool(re.fullmatch(pattern, s))


def errors_ber(y_true, y_pred, positions = 'default'):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    myOtherTensor = torch.ne(torch.round(y_true), torch.round(y_pred)).float()
    if positions == 'default':
        res = sum(sum(myOtherTensor))/(myOtherTensor.shape[0]*myOtherTensor.shape[1])
    else:
        res = torch.mean(myOtherTensor, dim=0).type(torch.FloatTensor)
        for pos in positions:
            res[pos] = 0.0
        res = torch.mean(res)
    return res

def errors_ber_list(y_true, y_pred):
    block_len = y_true.shape[1]
    y_true = y_true.view(y_true.shape[0], -1)
    y_pred = y_pred.view(y_pred.shape[0], -1)

    myOtherTensor = torch.ne(torch.round(y_true), torch.round(y_pred))
    res_list_tensor = torch.sum(myOtherTensor, dim = 1).type(torch.FloatTensor)/block_len

    return res_list_tensor


def errors_ber_pos(y_true, y_pred, discard_pos = []):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    myOtherTensor = torch.ne(torch.round(y_true), torch.round(y_pred)).float()

    tmp =  myOtherTensor.sum(0)/myOtherTensor.shape[0]
    res = tmp.squeeze(1)
    return res

def code_power(the_codes):
    the_codes = the_codes.cpu().numpy()
    the_codes = np.abs(the_codes)**2
    the_codes = the_codes.sum(2)/the_codes.shape[2]
    tmp =  the_codes.sum(0)/the_codes.shape[0]
    res = tmp
    return res

def errors_bler(y_true, y_pred, positions = 'default'):

    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    decoded_bits = torch.round(y_pred)
    X_test       = torch.round(y_true)
    tp0 = (abs(decoded_bits-X_test)).view([X_test.shape[0],X_test.shape[1]])
    tp0 = tp0.cpu().numpy()

    if positions == 'default':
        bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])
    else:
        for pos in positions:
            tp0[:, pos] = 0.0
        bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])

    return bler_err_rate

# note there are a few definitions of SNR. In our result, we stick to the following SNR setup.
def snr_db2sigma(train_snr):
    return 10**(-train_snr*1.0/20)

def snr_sigma2db(train_sigma):
    try:
        return -20.0 * math.log(train_sigma, 10)
    except:
        return -20.0 * torch.log10(train_sigma)