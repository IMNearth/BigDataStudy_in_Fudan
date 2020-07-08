# -*- coding: utf-8 -*-
"""
@create_time : 2020-04-07 13:06
@environment : python 3.6
@author      : zhangjiwen
@file        : mapper
"""

import sys
import re
import jieba
from collections import Counter


pattern1 = r"<[a-zA-Z]*>"  # <>
pattern2 = r">[^a-z]*"  # ><
pattern3 = r"\".+?\""

contents = []
titles = []

for line in sys.stdin:
    result = re.search(pattern1, line)
    if result is not None:
        typo = result.group()[1:-1]
        _text = re.search(pattern2, line).group()[1:-2]
        if typo == "content":
            contents.append(_text)
        elif typo == "contenttitle":
            _text = _text[:-1] if (_text != "" and _text[-1] == "（") else _text
            titles.append(_text)


for i in range(len(titles)):
    title, doc = titles[i], contents[i]
    seg_list = list(jieba.cut(doc, cut_all=True))
    counter = Counter(seg_list)
    total_word_frequency = sum(counter.values())
    for pos, word in enumerate(set(seg_list)):
        print("{}@{}\t{}".format(word, counter[word], title))
        # counter[word]*100/float(total_word_frequency)