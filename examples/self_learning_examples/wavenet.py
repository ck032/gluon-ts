# _*_ coding:utf-8 _*_

# @Author      : chenkai<chenkai15@geely.com>
# @Created Time: 2020/8/1 下午3:39
# @File        : wavenet.py

import tushare as ts
from tushare.pro import data_pro

# my_token = '600cc23991e8c96cac0ea4ed662d46a5dff74243638ea6ffe6220601'
# ts.set_token(token=my_token)

data = data_pro.pro_bar(ts_code='002415')
print(data)
