# -*- coding: utf-8 -*-
# @Time    : 18-3-6 3:20 PM
# @Author  : zengzihua@huya.com
# @FileName: data_filter.py
# @Software: PyCharm

from src.networks import network_mv2_cpm, network_mv2_hourglass

## Used to import models for CPM and hourglass models
def get_network(type, input, trainable=True, scale = 1):
    if type == 'mv2_cpm':
        net, loss = network_mv2_cpm.build_network(input, trainable, scale)
    elif type == "mv2_hourglass":
        net, loss = network_mv2_hourglass.build_network(input, trainable)
    return net, loss
