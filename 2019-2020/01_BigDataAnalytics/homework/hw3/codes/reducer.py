# -*- coding: utf-8 -*-
"""
@create_time : 2020-04-07 15:44
@environment : python 3.6
@author      : zhangjiwen
@file        : reducer
"""
import sys
import re
from collections import defaultdict

out_dict = defaultdict(list)

for line in sys.stdin:
    key, value = line.split("\t")
    word, count = key.split("@")
    out_dict[word].append(value[:-1]+"@"+count)

for word, value_list in out_dict.items():
    titles = "<SEP>".join(value_list)
    print("{}\t{}".format(word, titles))