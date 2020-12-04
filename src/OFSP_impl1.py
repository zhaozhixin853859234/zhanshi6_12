import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, metrics, tree, linear_model
from sklearn.naive_bayes import GaussianNB

# 实现ofsp方法，参数data为数据集，view为初始化视图，字典保存,
# model为特征选择时使用的学习器
def ofsp(data,view):
    #feature_index = data.shape[1]

    # 存储特征加入某个视图后验证准确度,用字典存储
    view_names = []
    for key in view.keys():
        view_names.append(key)
    # 初始化论文中Ai=0
    view_valid_acc = dict(zip(view_names, [999 for i in range(len(view))]))
    # 存储视图加入特征后前后准确度差值
    view_acc_sub = dict(zip(view_names, [0 for i in range(len(view))]))
    # 初始化论文中A_temp=0
    view_valid_acc_temp = dict(zip(view_names, [0 for i in range(len(view))]))

    # 分割数据集
    data = data.values
    feature_all = data[::, 0:len(data[0]) - 1]
    label = data[::, len(data[0]) - 1]
    train_all_features, test_all_features, train_all_labels, test_all_labels = \
        train_test_split(feature_all, label, test_size=0.1, random_state=10)

    for i in range(len(data[0])-1):
        # 轮流将每个特征加入到每个视图内，考察准确度变化
        for view_keys in view_names:
            # 将特征加入视图
            view[view_keys].append(i)

            # 视图上的训练数据和标签
            train_feature_view = train_all_features[::, view[view_keys]]
            train_label_view = train_all_labels
            test_features_view = test_all_features[::, view[view_keys]]
            test_label_view = test_all_labels

            # 在视图上用不同分类器训练模型,每次要在视图上重新训练分类器
            #clf = svm.SVC()
            #clf = GaussianNB()
            #clf = KNeighborsClassifier(n_neighbors=3)
            #clf = tree.DecisionTreeClassifier(criterion='entropy')
            clf = linear_model.LinearRegression()
            clf.fit(train_feature_view, train_label_view)
            # 视图上预测结果
            predicted_view = clf.predict(test_features_view)
            acc_view = mean_squared_error(test_label_view, predicted_view)

            # 将视图上预测结果存储到字典中,先判断准确度是否有提升，如果没有，特征就不加入该视图
            acc_improve = acc_view - view_valid_acc[view_keys]

            view_valid_acc_temp[view_keys] = acc_view
            view_acc_sub[view_keys] = acc_improve

            # 打印各个视图上准确度信息
            print("加入第", i, "个特征后视图", view_keys, "为：", view[view_keys], "该视图上准确度为：", acc_view)
            # 将特征从视图内移除
            view[view_keys].remove(i)

        # flag记录加入特征后准确度提升是否有负数
        flag = False
        for values in view_acc_sub.values():
            if values <= 0:
                flag = True

        if flag:
            # 找出前后准确度差值最大的视图，将特征加入该视图
            max_acc_improve_view = min(view_acc_sub, key=view_acc_sub.get)
            print("加入第",i,"个特征后准确度提升情况：",view_acc_sub)
            print("准确度最高的视图为", max_acc_improve_view, ",将第", i, "个特征加入该视图")
            view[max_acc_improve_view].append(i)
            view_valid_acc[max_acc_improve_view] = view_valid_acc_temp[max_acc_improve_view]
            print("本次迭代结果为", view)

        else:
            # 找出前后准确度差值最大的视图，将特征加入该视图
            print("加入第",i,"个特征后准确度提升情况：",view_acc_sub)
            print("加入第",i,"个特征后，所有视图上准确度提升为负，故舍弃该特征，不加入任何视图")

    return view

if __name__ == '__main__':
    df = pd.read_csv("../data/experiment_data/1791269877_train.csv", header=None)
    viewDict = {"V1": [], "V2": [], "V3": []}
    view = ofsp(df, viewDict)
    # 保存视图
    # 将最终视图构造结果存为json文件
    jsObj = json.dumps(view)
    view_save_path = str("../data/featureClusterResult/OFSP/ofsp_res_LinearModel.json")
    fileObject = open(view_save_path, 'w')
    fileObject.write(jsObj)
    fileObject.close()