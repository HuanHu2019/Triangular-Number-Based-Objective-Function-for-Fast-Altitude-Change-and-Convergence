# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 15:18:40 2023

@author: Administrator
"""

import sys
import os
import re


curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加路径到系统路径

name=os.path.basename(__file__)
print(name)

#去掉文件后缀，只要文件名称
# name=os.path.basename(__file__).split(".")[0]
# print(name)


your_daddad= os.path.basename(curr_path)

d = re.findall(r"s_(.+?)",your_daddad)
#d = int(d[0]) 
d = int(d[0]) 
print('d',d)