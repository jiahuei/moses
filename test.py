# -*- coding: utf-8 -*-
"""
Created on 13 Apr 2020 22:54:01

@author: jiahuei
"""
import os
import pandas as pd

pjoin = os.path.join

data = ['a', 'ab', 'abcd', 'abc']
data.sort(key=len)
print(data)
