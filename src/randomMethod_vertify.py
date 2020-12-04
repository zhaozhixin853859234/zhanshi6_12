import pandas as pd
import random
import json
# 设置函数
# listTemp 为列表平分后每份列表的的个数n
def divide_equally(listTemp, n):
    res = []
    step = int(len(listTemp)/n)
    for i in range(n-1):
        res.append(listTemp[step*i:step*(i+1)])
    res.append(listTemp[step*(n-1):len(listTemp)])
    return res

if __name__ == '__main__':
    df = pd.read_csv("../data/experiment_data/1791269877_train.csv",header=None)
    num_feantures = df.shape[1]-1
    # 将特征索引存入数组中，然后随机从数组中选择特征索引放入对应视图中
    num_feantures_list = list(range(num_feantures))

    # 需要构造视图数量
    num_views = 3

    # 用字典存储视图,生成字典的键值
    view_names = []
    for i in range(1, num_views+1):
        view_names.append(str("V")+str(i))
    view_dict_random = dict(zip(view_names, [[] for i in range(num_views)]))

    # print(view_dict)
    # print(num_feantures_list)

    # 将所有特征索引随机加入到视图中
    while len(num_feantures_list) > 0:
        for i in view_names:
            if len(num_feantures_list) > 0:
                random_feature_index = random.choice(num_feantures_list)
                view_dict_random[i].append(random_feature_index)
                num_feantures_list.remove(random_feature_index)
            else:
                break

    # 存储到json中
    jsObj = json.dumps(view_dict_random)
    view_save_path = str("../data/featureClusterResult/random_subspace/randomMethod_")+str(num_views)\
                     + str("_views_discretion.json")
    fileObject = open(view_save_path, 'w')
    fileObject.write(jsObj)
    fileObject.close()

    # 连续平分特征
    num_feantures_list = list(range(num_feantures))
    res = divide_equally(num_feantures_list,num_views)
    view_dict_divide_equally = dict(zip(view_names, res))
    # print(view_dict_divide_equally)
    # 存储到json中
    jsObj = json.dumps(view_dict_divide_equally)
    view_save_path = str("../data/featureClusterResult/feature_junfen/divide_equally_")+str(num_views)\
                     + str("_views_discretion.json")
    fileObject = open(view_save_path, 'w')
    fileObject.write(jsObj)
    fileObject.close()