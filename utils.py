__author__ = 'yihanjiang'
import torch
import numpy as np
import math
from typing import List

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

def str_completion(args_len, sentences):
    """将长度不足的句子补全"""
    for i in range(len(sentences)):
        if len(sentences[i]) < args_len:
            sentences[i] += "*" * (args_len - len(sentences[i]))

    return sentences

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