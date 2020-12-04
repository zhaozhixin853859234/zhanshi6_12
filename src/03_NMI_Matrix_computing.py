import pandas as pd
import numpy as np
from sklearn import metrics

if __name__ == '__main__':
    df = pd.read_csv("../data/experiment_data/1791269877_train.csv", header=None)
    # df = df.drop([46],axis=1)
    # df = pd.DataFrame(df)
    # df.to_csv("../data/experiment_data/1791269877_train_temp.csv",header=None,index=None)
    # df = pd.read_csv("../data/experiment_data/1791269877_train_temp.csv", header=None)
    #df.reset_index(drop=True,inplace=True)
    print(df)
    # df.shape[0]返回行数,df.shape[1]返回列数
    # df1 = df.drop([df.shape[1]-1],axis=1)
    num_attributes = df.shape[1]
    NMI_matrix = np.zeros((num_attributes, num_attributes))
    MI_matrix = np.zeros((num_attributes, num_attributes))
    for i in range(num_attributes):
        attr1 = list(df[i])
        print("正在计算第", i, "个特征与其他特征之间互信息")
        for j in range(i, num_attributes):
            attr2 = list(df[j])
            NMI = metrics.normalized_mutual_info_score(attr1, attr2)
            MI = metrics.mutual_info_score(attr1, attr2)
            NMI_matrix[i][j] = NMI
            NMI_matrix[j][i] = NMI
            MI_matrix[i][j] = MI
            MI_matrix[j][i] = MI
    NMI_info = pd.DataFrame(NMI_matrix, columns=None, index=None)
    MI_info = pd.DataFrame(MI_matrix, columns=None, index=None)
    # 把最后一行删除，最后一行是标签与各个特征的互信息
    NMI_info = NMI_info.drop(NMI_info.shape[0]-1, axis=0)
    MI_info = MI_info.drop(MI_info.shape[0] - 1, axis=0)

    NMI_info.to_csv("../data/NMI_info_withLabel.csv", index=None)
    MI_info.to_csv("../data/MI_info_withLabel.csv",  index=None)
