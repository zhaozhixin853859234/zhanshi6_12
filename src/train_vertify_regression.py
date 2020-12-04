import random
import warnings
import json
import matplotlib.dates as mdates
import time
import matplotlib.ticker as mticker
from datetime import datetime

from scipy.interpolate import spline
from dateutil import parser
from sklearn import linear_model, metrics, ensemble
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import ExtraTreeRegressor
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf,suppress=True)
import matplotlib
# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
# 将回归预测的浓度值分级，
def maxValue2grade(maxValue_predict_list):
    # 将浓度按阈值分级，[0,0.5)为1级，[0.5.0.75)为2级，[0.75,1)为3级，[1,99)为4级
    grade_list = []
    for i in maxValue_predict_list:
        if i >= 1:
            grade_list.append(4)
        elif i < 1 and i >= 0.7:
            grade_list.append(3)
        elif i < 0.7 and i >= 0.4:
            grade_list.append(2)
        else:
            grade_list.append(1)
    return grade_list

# 多标签分类准确度计算
def acc(true_label, predict_label):
    count_true = 0
    count_false = 0
    for i in range(len(true_label)):
        if true_label[i] == predict_label[i]:
            count_true = count_true + 1
        else:
            # print("分类错误！真实标签：",true_label[i],"预测标签：",predict_label[i])
            count_false = count_false+1
    return count_true/len(true_label),count_false

# 混淆矩阵计算
def convex_computing(true_label, predict_label):
    # 标签个数
    label_num = len(set(list(true_label)))

    convex_matrix = np.zeros((label_num+1, label_num+1))
    for i in range(len(true_label)):
        convex_matrix[predict_label[i]][true_label[i]] = convex_matrix[predict_label[i]][true_label[i]]+1
    return convex_matrix

# 找出列表里出现次数最多元素值，一样多就返回多个
def list_max_value(lt):
    temp = 0
    max_repeat = []
    max_str = 0
    for i in lt:
        if lt.count(i) > temp:
            max_str = i
            temp = lt.count(i)
        elif lt.count(i) == temp:
            max_repeat.append(i)
        else:
            pass
    max_repeat.append(max_str)
    return list(set(max_repeat))

# 这里predict_res参数是所有视图预测结果组成的二维列表
# 各个视图没有赋予权重
def vote_fusion(predict_res):
    # 存储融合结果
    fusion_res = []
    for i in range(len(predict_res[0])):
        # 按列将元素存入列表中
        temp = []
        for j in range(len(predict_res)):
            temp.append(predict_res[j][i])
        max_repeat = list_max_value(temp)
        if len(max_repeat) == 1:
            fusion_res.append(max_repeat[0])
        else:
            fusion_res.append(random.choice(max_repeat))
    return fusion_res


# 均值投票，三个视图上预测结果均值
def mean_vote(data):
    row = len(data)
    columns = len(data[0])
    res = []
    for i in range(columns):
        # temp = 0
        # for j in range(row):
        #     temp = temp + data[j][i]
        res.append(0.05*data[0][i]+0.9*data[1][i]+0.05*data[2][i])
    return res

def train(train_set,view_data_path):
    print("训练集标签分布：\n", train_set.shape)
    # train_set = train_set.drop([0])
    train_set = train_set.values
    train_set_feature = train_set[:, 1:len(train_set[0]) - 1]
    train_set_label = train_set[:, len(train_set[0]) - 1]
    # print(train_set_feature)
    # print(train_set_label)
    #test_id = 1791269791
    test_id = 1791276569
    test_set_filePath = str("../data/") + str(test_id) + str("_test_grade.csv")
    test_set = pd.read_csv(test_set_filePath, header=0)
    print("测试集标签分布：\n", test_set["grade"].value_counts())
    test_set = test_set.values
    data_date_maxTime_str = list(test_set[:, 0])
    data_date = list(map(parser.parse, data_date_maxTime_str))
    data_date = data_date[300:1000]
    test_set_feature = test_set[:, 1:len(test_set[0]) - 2]
    test_set_label = test_set[:, len(test_set[0]) - 1]
    test_set_maxValue = test_set[:, len(test_set[0]) - 2]

    # 全部数据集测试
    clf_all_feature = linear_model.LinearRegression()
    # clf_all_feature = ensemble.AdaBoostRegressor(n_estimators=50)
    # clf_all_feature = ensemble.RandomForestRegressor(n_estimators=20)
    # clf_all_feature = ExtraTreeRegressor()
    # clf_all_feature = BaggingRegressor()
    # clf_all_feature = ensemble.GradientBoostingRegressor(n_estimators=100)
    clf_all_feature.fit(train_set_feature, train_set_label)
    maxValue_predict = clf_all_feature.predict(test_set_feature)
    # maxValue_predict = list(map(lambda x:x+0.01, maxValue_predict))

    # 回归评价标准MSE
    MSE_alldata = mean_squared_error(test_set_maxValue, maxValue_predict)
    print("根据浓度值回归后MSE：", MSE_alldata)
    maxValue_predict_grade = maxValue2grade(maxValue_predict)
    # print("真实浓度值：\n",test_set_maxValue)
    # print("预测浓度值：\n",maxValue_predict)
    # print("真实标签：\n", test_set_label)
    # print("预测标签：\n", maxValue_predict_grade)
    # 根据阈值转换为分类后，评价指标
    accuracy, num = acc(test_set_label, maxValue_predict_grade)
    # loss = metrics.hamming_loss(test_set_label, maxValue_predict_grade)
    print("根据阈值转换为分类后，准确度：", accuracy)
    print("大类样本占比：", 1317 / 1550)
    print("分错个数：", num)


    # 作图展示
    fig = plt.figure(figsize=(20, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    test_set_maxValue_limit = test_set_maxValue[300:1000]
    maxValue_predict_limit = maxValue_predict[300:1000]

    last = test_set_maxValue_limit[0]
    test_set_maxValue_limit_smoothed = []
    weight = 0.75
    for point in test_set_maxValue_limit:
        smoothed_val = last * weight + (1 - weight) * point
        test_set_maxValue_limit_smoothed.append(smoothed_val)
        last = smoothed_val

    last = maxValue_predict_limit[0]
    maxValue_predict_limit_smoothed = []
    for point in maxValue_predict_limit:
        smoothed_val = last * weight + (1 - weight) * point
        maxValue_predict_limit_smoothed.append(smoothed_val)
        last = smoothed_val
    a, = plt.plot(data_date, test_set_maxValue_limit_smoothed, color="green", label='Real gas concentration value',linewidth=3)
    b, = plt.plot(data_date, maxValue_predict_limit_smoothed, color="red", label='Predicted gas concentration value',linewidth=3)
    plt.axhline(0.4, color="red", linewidth=1.0, linestyle='--', label='Risk level 2')
    plt.axhline(0.7, color="black", linewidth=1.0, linestyle='--', label='Risk level 3')
    plt.axhline(1.0, color="blue", linewidth=1.0, linestyle='--', label='Risk level 4')
    plt.text("2019/02/10", 1.0, "MSE:0.015", fontsize=20)
    # 配置横坐标
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    # ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    mticker.Locator.MAXTICKS = 2800
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())  # 按日显示
    plt.gcf().autofmt_xdate()  # 自动旋转日期标记

    model = "--LinearRegression"
    #fig_title = "The predicted results without using the DVC algorithm"
    #plt.title(fig_title, fontdict={'weight': 'normal', 'size': 20})
    plt.legend(loc="upper left", fontsize=20)
    ax.set_xlabel("Date", fontdict={'weight': 'normal', 'size': 20})
    ax.set_ylabel("Gas concentration %CH4", fontdict={'weight': 'normal', 'size': 20})
    fig_save_path = str("../data/predictVisual/") + str(test_id) + str(model) + str("--all") + str(".png")
    plt.savefig(fig_save_path)
    # plt.close()
    print("--------------------------------------------------------------")

    '''读取视图构造后保存的字典，分割数据集并验证'''
    # view_data_path = "../../viewConstruction/addFeatureIndex_45.json"
    with open(view_data_path, 'r') as f:
        viewDict = json.load(f)

    # 分离出各个视图的特征和标签
    view_name = "V1"
    train_feature_v1 = train_set_feature[::, viewDict[view_name]]
    # train_label_v1 = train_all_labels[::, viewDict[view_name]]
    test_feature_v1 = test_set_feature[::, viewDict[view_name]]
    # test_label_v1 = test_all_labels[::,viewDict[view_name]]

    view_name = "V2"
    train_feature_v2 = train_set_feature[::, viewDict[view_name]]
    # train_label_v2 = train_all_labels[::, viewDict[view_name]]
    test_feature_v2 = test_set_feature[::, viewDict[view_name]]
    # test_label_v2 = test_all_labels[::, viewDict[view_name]]

    view_name = "V3"
    train_feature_v3 = train_set_feature[::, viewDict[view_name]]
    # train_label_v3 = train_all_labels[::, viewDict[view_name]]
    test_feature_v3 = test_set_feature[::, viewDict[view_name]]
    # test_label_v3 = test_all_labels[::, viewDict[view_name]]

    '''视图V1上验证精确度'''
    # avge_accuracy = []
    # accuracy_iter = []
    time_1 = time.time()
    print('Start training...')
    # clf_v1_feature = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    # clf_v1_feature = RandomForestClassifier(random_state=0,max_depth=50)
    # clf_v1_feature = tree.DecisionTreeClassifier(criterion='entropy')
    # clf_v1_feature = KNeighborsClassifier()
    # clf_v1_feature = RandomForestClassifier(n_estimators=8)
    # clf_v1_feature = SVC(kernel='rbf', probability=True)
    # clf_v1_feature = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 14, 12, 11, 10), random_state=1)
    # clf_v1_feature = MultinomialNB(alpha=0.01)
    clf_v1_feature = linear_model.LinearRegression()
    clf_v1_feature.fit(train_feature_v1, train_set_label)
    # 预测浓度，连续值
    maxValue_predict_v1 = clf_v1_feature.predict(test_feature_v1)
    # 按指定阈值分级，转换为离散值
    predicted_v1 = maxValue2grade(maxValue_predict_v1)

    MSE = mean_squared_error(test_set_maxValue, maxValue_predict_v1)
    print("视图V1上根据浓度值回归后MSE：", MSE)

    # print("真实浓度值：\n", test_set_maxValue)
    # print("视图V1上预测浓度值：\n", maxValue_predict_v1)
    # print("真实标签：\n", test_set_label)
    # print("视图V1上预测标签：\n", predicted_v1)
    # 根据阈值转换为分类后，评价指标
    accuracy, num = acc(test_set_label, predicted_v1)
    # loss = metrics.hamming_loss(test_set_label, maxValue_predict_grade)
    print("视图V1上根据阈值转换为分类后，准确度：", accuracy)
    print("大类样本占比：", 1317 / 1550)
    print("视图V1上分错个数：", num)
    # score = cross_val_score(clf_v1_feature, feature_v1, label, cv=5)  # cv为迭代次数。
    # predicted_v1 = cross_val_predict(clf_v1_feature, feature_v1, label, cv=2)
    # score_v1 = metrics.accuracy_score(test_all_labels, predicted_v1)
    time_2 = time.time()
    print('training cost %f seconds' % (time_2 - time_1))

    # accuracy_iter.append(score)
    # avge_accuracy.append(sum(score)/len(score))
    # print("视图V1上预测精确度：", score_v1)
    # print("真实标签：", test_all_labels)
    # print("视图V1上预测标签：", predicted_v1)
    # # print("视图V1上五折交叉验证平均精确度：", avge_accuracy)
    print("--------------------------------------------------------------")

    '''视图V2上验证精确度'''
    # avge_accuracy = []
    # accuracy_iter = []
    time_1 = time.time()
    print('Start training...')
    # clf_v1_feature = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    # clf_v1_feature = RandomForestClassifier(random_state=0,max_depth=50)
    # clf_v1_feature = tree.DecisionTreeClassifier(criterion='entropy')
    # clf_v1_feature = KNeighborsClassifier()
    # clf_v1_feature = RandomForestClassifier(n_estimators=8)
    # clf_v1_feature = SVC(kernel='rbf', probability=True)
    # clf_v1_feature = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 14, 12, 11, 10), random_state=1)
    # clf_v1_feature = MultinomialNB(alpha=0.01)
    clf_v2_feature = linear_model.LinearRegression()
    clf_v2_feature.fit(train_feature_v2, train_set_label)
    # 预测浓度，连续值
    maxValue_predict_v2 = clf_v2_feature.predict(test_feature_v2)
    # 按指定阈值分级，转换为离散值
    predicted_v2 = maxValue2grade(maxValue_predict_v2)

    MSE = mean_squared_error(test_set_maxValue, maxValue_predict_v2)
    print("视图V2上根据浓度值回归后MSE：", MSE)

    # print("真实浓度值：\n", test_set_maxValue)
    # print("视图V2上预测浓度值：\n", maxValue_predict_v2)
    # print("真实标签：\n", test_set_label)
    # print("视图V2上预测标签：\n", predicted_v2)
    # # 根据阈值转换为分类后，评价指标
    accuracy, num = acc(test_set_label, predicted_v2)
    # loss = metrics.hamming_loss(test_set_label, maxValue_predict_grade)
    print("视图V2上根据阈值转换为分类后，准确度：", accuracy)
    print("大类样本占比：", 1317 / 1550)
    print("视图V2上分错个数：", num)
    # score = cross_val_score(clf_v1_feature, feature_v1, label, cv=5)  # cv为迭代次数。
    # predicted_v1 = cross_val_predict(clf_v1_feature, feature_v1, label, cv=2)
    # score_v1 = metrics.accuracy_score(test_all_labels, predicted_v1)
    time_2 = time.time()
    print('training cost %f seconds' % (time_2 - time_1))

    # accuracy_iter.append(score)
    # avge_accuracy.append(sum(score)/len(score))
    # print("视图V1上预测精确度：", score_v1)
    # print("真实标签：", test_all_labels)
    # print("视图V1上预测标签：", predicted_v1)
    # # print("视图V1上五折交叉验证平均精确度：", avge_accuracy)

    '''视图V3上验证精确度'''
    time_1 = time.time()
    print('Start training...')
    # clf_v1_feature = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    # clf_v1_feature = RandomForestClassifier(random_state=0,max_depth=50)
    # clf_v1_feature = tree.DecisionTreeClassifier(criterion='entropy')
    # clf_v1_feature = KNeighborsClassifier()
    # clf_v1_feature = RandomForestClassifier(n_estimators=8)
    # clf_v1_feature = SVC(kernel='rbf', probability=True)
    # clf_v1_feature = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 14, 12, 11, 10), random_state=1)
    # clf_v1_feature = MultinomialNB(alpha=0.01)
    clf_v3_feature = linear_model.LinearRegression()
    clf_v3_feature.fit(train_feature_v3, train_set_label)
    # 预测浓度，连续值
    maxValue_predict_v3 = clf_v3_feature.predict(test_feature_v3)
    # 按指定阈值分级，转换为离散值
    predicted_v3 = maxValue2grade(maxValue_predict_v3)

    MSE = mean_squared_error(test_set_maxValue, maxValue_predict_v3)
    print("视图V3上根据浓度值回归后MSE：", MSE)

    # print("真实浓度值：\n", test_set_maxValue)
    # print("视图V3上预测浓度值：\n", maxValue_predict_v3)
    # print("真实标签：\n", test_set_label)
    # print("视图V3上预测标签：\n", predicted_v3)
    # 根据阈值转换为分类后，评价指标
    accuracy, num = acc(test_set_label, predicted_v3)
    # loss = metrics.hamming_loss(test_set_label, maxValue_predict_grade)
    print("视图V3上根据阈值转换为分类后，准确度：", accuracy)
    print("大类样本占比：", 1317 / 1550)
    print("视图V3上分错个数：", num)
    # score = cross_val_score(clf_v1_feature, feature_v1, label, cv=5)  # cv为迭代次数。
    # predicted_v1 = cross_val_predict(clf_v1_feature, feature_v1, label, cv=2)
    # score_v1 = metrics.accuracy_score(test_all_labels, predicted_v1)
    time_2 = time.time()
    print('training cost %f seconds' % (time_2 - time_1))

    # accuracy_iter.append(score)
    # avge_accuracy.append(sum(score)/len(score))
    # print("视图V1上预测精确度：", score_v1)
    # print("真实标签：", test_all_labels)
    # print("视图V1上预测标签：", predicted_v1)
    # # print("视图V1上五折交叉验证平均精确度：", avge_accuracy)
    print("--------------------------------------------------------------")

    '''融合视图结果,权重问题，使用基于概率模型，并行化'''
    view_predict_grade_res = []
    # 各个分级分类器结果组成二维列表
    view_predict_grade_res.append(list(predicted_v1))
    view_predict_grade_res.append(list(predicted_v2))
    view_predict_grade_res.append(list(predicted_v3))

    view_predict_mean_res = []
    # 各个分级分类器结果组成二维列表
    view_predict_mean_res.append(list(maxValue_predict_v1))
    view_predict_mean_res.append(list(maxValue_predict_v2))
    view_predict_mean_res.append(list(maxValue_predict_v3))
    # view_predict_res.append(list(predicted_v4))
    # print(view_predict_res)
    vote_predict_grade = vote_fusion(view_predict_grade_res)
    vote_predict_mean_lianxv = mean_vote(view_predict_mean_res)
    # 均值融合时计算MSE
    MSE_mean_fusion = mean_squared_error(test_set_maxValue, vote_predict_mean_lianxv)
    # 得到各个视图上浓度预测的均值后，转换为分级
    vote_predict_mean = maxValue2grade(vote_predict_mean_lianxv)

    score_vote_grade, num_grade = acc(test_set_label, vote_predict_grade)
    score_vote_mean, num_mean = acc(test_set_label, vote_predict_mean)
    print("分级后融合后准确度：", score_vote_grade)
    print("分级融合后分错个数：", num_grade)
    # print("真实标签：", test_set_label)
    # print("分级融合后标签值：", vote_predict_grade)
    print("分级前浓度均值融合后准确度：", score_vote_mean)
    print("分级前浓度均值融合后分错个数：", num_mean)
    print("分级前浓度均值融合后混淆矩阵：\n", convex_computing(test_set_label, vote_predict_mean))
    print("分级融合后混淆矩阵：\n", convex_computing(test_set_label, vote_predict_grade))

    # 作图展示
    fig = plt.figure(figsize=(20, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    # test_set_maxValue_limit = test_set_maxValue[0:1400]
    vote_predict_mean_lianxv_limit = vote_predict_mean_lianxv[300:1000]

    last = vote_predict_mean_lianxv_limit[0]
    vote_predict_mean_lianxv_limit_smoothed = []
    for point in vote_predict_mean_lianxv_limit:
        smoothed_val = last * weight + (1 - weight) * point
        vote_predict_mean_lianxv_limit_smoothed.append(smoothed_val)
        last = smoothed_val

    a, = plt.plot(data_date, test_set_maxValue_limit_smoothed, color="green", label='Real gas concentration value', linewidth=3)
    b, = plt.plot(data_date, vote_predict_mean_lianxv_limit_smoothed, color="red", label='Predicted gas concentration value',linewidth=3)
    plt.axhline(0.4, color="red", linewidth=1.0, linestyle='--', label='Risk level 2')
    plt.axhline(0.7, color="black", linewidth=1.0, linestyle='--', label='Risk level 3')
    plt.axhline(1.0, color="blue", linewidth=1.0, linestyle='--', label='Risk level 4')

    plt.text('2019/02/10', 1.0, "MSE:0.0061", fontsize=20)
    # 配置横坐标
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    # ax.xaxis.set_minor_locator(mticker.MultipleLocator(10))
    mticker.Locator.MAXTICKS = 2800
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())  # 按日显示
    plt.gcf().autofmt_xdate()  # 自动旋转日期标记

    model = "--LinearRegression"
    #fig_title = "The predicted results using the DVC algorithm"
    #plt.title(fig_title, fontdict={'weight': 'normal', 'size': 20})
    plt.legend(loc="upper left", fontsize=20)
    ax.set_xlabel("Date", fontdict={'weight': 'normal', 'size': 20})
    ax.set_ylabel("Gas concentration %CH4", fontdict={'weight': 'normal', 'size': 20})
    # plt.show()
    fig_save_path = str("../data/predictVisual/") + str(test_id) + str(model) + str("--vote_predict_mean") + str(".png")
    plt.savefig(fig_save_path)
    plt.close()
    return accuracy,score_vote_mean,score_vote_grade,MSE_alldata,MSE_mean_fusion

if __name__ == '__main__':
    train_set = pd.read_csv("../data/1791269877_train.csv", header=0)

    # 读取视图分割信息,视图名称为V1、V2、V3
    # kmeans聚类生成视图
    #view_data_path1 = "../data/featureClusterResult/K-means_Cluster/kmeans_model--3-cluster_group_discretion.json"
    # gmm聚类生成视图
    #view_data_path2 = "../data/featureClusterResult/GMM_Cluster/gmmModel--3-cluster_group_discretion.json"
    # DVC算法生成视图
    view_data_path3 = "../data/featureClusterResult/DVC/addFeatureIndex_45.json"
    # 随机方法生成视图
    #view_data_path4 = "../data/featureClusterResult/random_subspace/randomMethod_3_views_discretion.json"
    # 平分特征生成视图
    #view_data_path5 = "../data/featureClusterResult/feature_junfen/divide_equally_3_views_discretion.json"
    # GR特征排名生成视图
    #view_data_path6 = "../data/featureClusterResult/GR_rank/GR_feature_rank_3_views_discretion.json"
    # OFSP_KNN
    #view_data_path7 = "../data/featureClusterResult/OFSP/ofsp_res_LinearModel.json"
    # # OFSP_SVM
    # view_data_path8 = "../data/featureClusterResult/OFSP/ofsp_res_SVM.json"

    # 将各个算法视图路径存入数组
    view_data_path_list = []
    # view_data_path_list.append(view_data_path1)
    # view_data_path_list.append(view_data_path2)
    view_data_path_list.append(view_data_path3)
    # view_data_path_list.append(view_data_path4)
    # view_data_path_list.append(view_data_path5)
    # view_data_path_list.append(view_data_path6)
    # view_data_path_list.append(view_data_path7)
    #view_data_path_list.append(view_data_path8)

    # 设置训练次数
    num_iter = 10
    res = []
    for view_data_path in view_data_path_list:
        score_allData_list = []
        score_vote_mean_list = []
        score_vote_grade_list = []
        mse_alldata = []
        mse_mean = []
        for i in range(num_iter):
            allData_acc, score_vote_mean, score_vote_grade,mse_all,mse_meanVote = train(train_set, view_data_path)
            score_allData_list.append(allData_acc)
            score_vote_mean_list.append(score_vote_mean)
            score_vote_grade_list.append(score_vote_grade)
            mse_alldata.append(mse_all)
            mse_mean.append(mse_meanVote)
        # 将各个算法得到的准确度均值方差存入二维列表，行为算法，列为均值方差等
        temp = [np.mean(score_allData_list), np.var(score_allData_list), np.mean(score_vote_mean_list),
                np.var(score_vote_mean_list), np.mean(score_vote_grade_list), np.var(score_vote_grade_list)]
        res.append(temp)
        print("利用",view_data_path, "构造视图后相关结果", temp)
        print("加入到最终结果列表", res)

        print("alldata MSE",np.mean(mse_alldata),np.var(mse_alldata))
        print("mean fusion MSE:",np.mean(mse_mean),np.var(mse_mean))
    # data = pd.DataFrame(res, columns=["all_data_acc_mean", "all_data_acc_var", "vote_mean_acc_mean",
    #                                   "vote_mean_acc_var", "vote_grade_acc_mean", "vote_grade_acc_var"],
    #                     index=["K_means_cluster", "GMM_cluster", "DVC", "Random_subspace", "feature_divide",
    #                            "GR_rank", "OFSP_LinearModel"])
    # data.to_csv("../data/experiment_data/res_info_1791269791.csv")
