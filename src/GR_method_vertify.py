import pandas as pd
import numpy as np
from collections import deque
import json
import time


# 计算信息熵
def getEntropy(s):
    # 找到各个不同取值出现的次数
    if not isinstance(s, pd.core.series.Series):
        s = pd.Series(s)
    prt_ary = pd.groupby(s, by=s).count().values / float(len(s))
    return -(np.log2(prt_ary) * prt_ary).sum()

# 计算条件熵: 条件s1下s2的条件熵
def getCondEntropy(s1, s2):
    d = dict()
    for i in list(range(len(s1))):
        # 把条件和结论的取值对应起来：
        # 最后结果：{'X2': ['Y1', 'Y2', 'Y2', 'Y2'], 'X1': ['Y1', 'Y1']}
        # d.get(s1[i], [])：取得字典d的键值s1[i]对应的value，
        #print(d.get(s1[i], []))
        d[s1[i]] = d.get(s1[i], [])+[s2[i]]
        # print("ss", d)
    return sum([getEntropy(d[k]) * len(d[k]) / float(len(s1)) for k in d])

# 计算信息增益
def getEntropyGain(s1, s2):
    return getEntropy(s2) - getCondEntropy(s1, s2)

# 计算增益率
# 出现信息增益率为1的情况是因为连续属性没有重复取值
# 每个子集只有一个元素
# 信息增益准则对那些属性的取值比较多的属性有所偏好
def getEntropyGainRadio(s1, s2):
    return getEntropyGain(s1, s2) / getEntropy(s1)

# 基于GR的特征排名视图构造方法
# 参数feature_GR为各个特征的信息增益率，num_view为视图个数
def GR_feature_rank(feature_GR,num_view):
    # 特征的信息增益率逆序排列,并将特征存入列表中
    feature_index_list = []
    sort_GR_dict = sorted(feature_GR.items(), key=lambda x: x[1], reverse=True)
    print("sort_GR_dict", sort_GR_dict)
    for i in sort_GR_dict:
        feature_index_list.append(i[0])

    # 队列存储更加方便
    feature_index_que = deque(feature_index_list)
    # 根据视图个数num_view建立字典存储视图
    # 形式为{'V1': [], 'V2': [], 'V3': [],...}
    str_name = list(range(1, num_view+1))
    view_names = list(map(lambda x: "V"+str(x), str_name))
    view = dict(zip(view_names, [[] for i in range(num_view)]))

    flag = True
    while flag:
        for i in range(1,num_view+1):
            if len(feature_index_que) > 0:
                view[str("V")+str(i)].append(feature_index_que.popleft())
            else:
                flag=False
    return view
    # # 奇数情况和偶数情况特征加入视图顺序不同
    # num_iteration = 1
    # while len(feature_index_que) > 0:
    #     if num_iteration%2 != 0:
    #         for i in range(1, num_view+1):
    #             if len(feature_index_que) > 0:
    #                 view["V" + str(i)].append(feature_index_que.popleft())
    #             else:
    #                 return view
    #     else:
    #         for i in range(num_view, 0, -1):
    #             if len(feature_index_que) > 0:
    #                 view["V" + str(i)].append(feature_index_que.popleft())
    #             else:
    #                 return view
    #     num_iteration = num_iteration + 1

    # for i in range(0, len(feature_index_list), num_view):
    #     if len(feature_index_list) - i >= num_view:
    #         # 奇数情况和偶数情况特征加入视图顺序不同
    #         if num_iteration%2 != 0:
    #             for j in range(1, num_view+1):
    #                 view["V"+str(j)].append(feature_index_list[i+j-1])
    #         else:
    #             for j in range(1, num_view + 1):
    #                 view["V" + str(j)].append(feature_index_list[i + num_view - j])
    #     else:
    #         yushu = len(feature_index_list)%num_view
    #         if num_iteration%2==0:
    #             if le
    #
    #
    #     num_iteration += 1

if __name__ == '__main__':
    df = pd.read_csv("../data/experiment_data/1791269877_train.csv", header=None)
    feature_GR = []
    time_start = time.time()
    for i in range(df.shape[1]-1):
        print("-------------------------------------")
        print("正在计算第", i, "个特征的信息增益率")
        GR_value = getEntropyGainRadio(df[i], df[df.shape[1]-1])
        feature_GR.append(GR_value)
        print("第", i, "个特征的信息增益率为：", GR_value)
        print("计算完成")
        print("-------------------------------------")
    feature_index = list(range(df.shape[1]-1))

    # 字典存储每个特征的信息增益率
    dict_feature_GR = dict(zip(feature_index, feature_GR))
    print(dict_feature_GR)
    # 最终视图构造结果
    view_num = 3
    view_res = GR_feature_rank(dict_feature_GR, view_num)
    time_end = time.time()
    print("最终视图构造结果\n", view_res)
    print("GR_rank视图构造时间：", time_end-time_start)
    #     # 保存json送文件中
    jsObj = json.dumps(view_res)
    view_save_path = str("../data/featureClusterResult/GR_rank/GR_feature_rank_")+str(view_num)\
                     + str("_views_discretion.json")
    fileObject = open(view_save_path, 'w')
    fileObject.write(jsObj)
    fileObject.close()

