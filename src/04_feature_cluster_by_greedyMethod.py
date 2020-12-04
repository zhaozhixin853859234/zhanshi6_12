import pandas as pd
import matlab.engine
from previous_code import tools
import json
import random
import time

class greedyMethod:
    # 初始化参数，MI_info为特征的互信息矩阵,k为CCA提取视图间关联时投影向量取值个数,
    # num_view为视图个数,add_feature表示需要加入的特征
    def __init__(self, MI_info, k, view_dict, add_feature_index):
        self.MI_info = MI_info
        self.k = k
        self.view_dict = view_dict
        self.add_feature_index = add_feature_index

    # 计算评价视图的关联公式,共有两个部分：
    # 先计算视图内特征之间的关联性，充分冗余性特征
    # 参数view_name传入应为视图的主键名，其中对应的值要与MI_info的索引对应
    def computing_intra_view_correlation(self, view_name):
        # 获取视图字典里指定视图名对应的特征索引
        feature_index_in_view = self.view_dict[view_name]
        fearure_num = len(feature_index_in_view)

        # intra_view_confidence表示视图内置信度，作为结果返回
        intra_view_confidence = 0

        # 计算新加入特征与该视图内所有特征的互信息之和，该特征与标签的互信息
        for i in feature_index_in_view:
            intra_view_confidence = intra_view_confidence + self.MI_info[i][self.add_feature_index]
            intra_view_confidence = intra_view_confidence + self.MI_info[i][len(self.MI_info[0])-1]

        # 加上该特征与标签的互信息,和视图内其他特征与标签的互信息
        intra_view_confidence = intra_view_confidence + self.MI_info[self.add_feature_index][len(self.MI_info[0])-1]
        return intra_view_confidence/(2*fearure_num + 1)

    # 再计算两个视图之间的关联性，条件独立性特征
    def computing_inter_view_correlation(self, view1_key, view2_key):
        # 字典存储所有视图，先获取视图对应的特征索引
        view1_attr_index = self.view_dict[view1_key]
        view2_attr_index = self.view_dict[view2_key]

        # inter_view_confidence表示视图间置信度，作为结果返回
        inter_view_confidence = 0

        # 把两个视图内的特征的互信息存入矩阵中
        view1_feature = []
        view2_feature = []
        for i in view1_attr_index:
            view1_feature.append(self.MI_info[i].tolist())
        for j in view2_attr_index:
            view2_feature.append(self.MI_info[j].tolist())

        # 将二维列表转置并转化为matlab使用的矩阵
        view1_feature = matlab.double(tools.transpose(view1_feature))
        view2_feature = matlab.double(tools.transpose(view2_feature))

        # 启动matlab连接,调用CCA函数，计算出结果
        # eng = matlab.engine.start_matlab()
        A, B, r, U, V = eng.canoncorr(view1_feature, view2_feature, nargout=5)
        print("关系系数列表", r)
        # eng.exit()
        # print("视图间关系系数：", r)
        # 公式计算最后结果
        # r表示相关系数，按大小排序取前k个,分为三种情况：
        # 1、只有一个关系系数，此时r为一个浮点数
        # 2、关系系数个数大于1小于k时，此时r为一个二维数组，所有关系系数都取,并取均值
        # 3、关系系数个数大于等于k时，此时r为一个二维数组，关系系数取前k个，并取均值
        if isinstance(r, float):
            # inter_view_confidence = inter_view_confidence + (-math.log2(1 - r*r))
            inter_view_confidence = r
        elif len(r[0]) > self.k:
            for i in range(self.k):
                # inter_view_confidence = inter_view_confidence + (-math.log2(1 - r[0][i]*r[0][i]))
                # 不进行非线性变换，直接求得均值
                inter_view_confidence = inter_view_confidence + r[0][i]
                inter_view_confidence = inter_view_confidence/self.k
        else:
            for i in range(len(r[0])):
                # inter_view_confidence = inter_view_confidence + (-math.log2(1 - r[0][i] * r[0][i]))
                inter_view_confidence = inter_view_confidence + r[0][i]
                inter_view_confidence = inter_view_confidence/len(r[0])
        return inter_view_confidence

    # 贪心策略进行迭代优化,每次只加入一个特征，主函数中循环遍历所有特征
    def feature_divide_by_greedyMethod(self,viewDict):
        view_names = []
        for key in self.view_dict.keys():
            view_names.append(key)

        # res_list存储最终需要优化的值，即视图内置信度减去视图间置信度，用字典存储每个视图对应的指标值
        res_list = dict(zip(view_names, [[] for i in range(len(viewDict))]))

        # 第一个for循环是将特征逐个加入到每个视图中
        # 第二个for循环计算加入新特征的视图与其他视图的视图间置信度之和
        for i in range(len(viewDict)):
            # 视图间置信度
            cca_coefficient = 0

            # 将特征加入视图中（索引）
            viewDict[view_names[i]].append(self.add_feature_index)
            print("将第", self.add_feature_index, "个特征加入视图", view_names[i], "后，新的视图为：")
            print(viewDict)
            gm = greedyMethod(self.MI_info, k=5, view_dict=viewDict, add_feature_index=self.add_feature_index)

            # intra_view_value定义为视图内置信度的值
            intra_view_value = gm.computing_intra_view_correlation(view_names[i])
            print("视图", view_names[i], "视图内置信度为：", intra_view_value)

            # 计算加入新特征的视图与其他视图的视图间置信度之和
            for j in range(len(viewDict)):
                if i != j:
                    # r为cca计算出两个视图间关系系数，累加和cca_coefficient即为视图间置信度
                    r = gm.computing_inter_view_correlation(view_names[i], view_names[j])
                    print("视图", view_names[i], "和视图", view_names[j], "间相关系数均值为：", r)
                    cca_coefficient = cca_coefficient + r

            # sigmod进行非线性变换，到[0,1]区间内,两个指标相除，取最大值
            res_list[view_names[i]].append(
                tools.sigmod(intra_view_value) / tools.sigmod(cca_coefficient / (len(viewDict) - 1)))

            # 第一个for循环结束后,将该特征索引从视图中删除，下次循环加入下一个视图中
            viewDict[view_names[i]].remove(self.add_feature_index)

        # 选出特征在分别加入三个视图后。评价指标值最大的所在视图，
        # 并把该特征加入该视图，返回修改后的视图为更新后的视图
        max_score_key = max(res_list, key=res_list.get)
        viewDict[str(max_score_key)].append(self.add_feature_index)
        print("加入第", self.add_feature_index, "个特征后，各个视图评价指标值：", res_list)
        print("加入第", self.add_feature_index, "个特征后,视图", max_score_key, "指标值最大，将该特征加入该视图中")
        return viewDict

        # 这里传进去的数据应该为原始数据，而不是特征间的互信息
        # gm = greedyMethod(self.MI_info, k=5, view_dict=viewDict, add_feature_index=self.add_feature_index)
        #
        # 计算某个特征加入视图1后，评价指标的值：视图内置信度与视图间置信度之差
        # intra1 = gm.computing_intra_view_correlation(view_names[0])
        # viewDict[view_names[0]].append(self.add_feature_index)
        # gm_v1 = greedyMethod(self.MI_info, k=5, view_dict=viewDict, add_feature_index=self.add_feature_index)
        # print("将第", self.add_feature_index, "个特征加入视图", view_names[0], "后，新的视图为：")
        # print(viewDict)

        # inter12 = gm_v1.computing_inter_view_correlation(view_names[0], view_names[1])
        # inter13 = gm_v1.computing_inter_view_correlation(view_names[0], view_names[2])
        # inter14 = gm_v1.computing_inter_view_correlation(view_names[1], view_names[2])
        # viewDict[view_names[0]].remove(self.add_feature_index)
        # score[view_names[0]].append(intra1 - inter12)

        # 计算某个特征加入视图2后，评价指标的值：视图内置信度与视图间置信度之差
        # intra2 = gm.computing_intra_view_correlation(view_names[1])
        # viewDict[view_names[1]].append(self.add_feature_index)
        # gm_v2 = greedyMethod(MI_matrix, k=5, view_dict=viewDict, add_feature_index=self.add_feature_index)
        # print("将第", self.add_feature_index, "个特征加入视图", view_names[1], "后，新的视图为：")
        # print(viewDict)
        # inter12 = gm_v2.computing_inter_view_correlation(view_names[0], view_names[1])
        # # inter13 = gm_v2.computing_inter_view_correlation(view_names[0], view_names[2])
        # # inter23 = gm_v2.computing_inter_view_correlation(view_names[1], view_names[2])
        # viewDict[view_names[1]].remove(self.add_feature_index)
        # score[view_names[1]].append(intra1 - inter12)
        # score[view_names[1]].append(intra2 - inter12 - inter13 - inter23)

        # 计算某个特征加入视图3后，评价指标的值：视图内置信度与视图间置信度之差
        # intra3 = gm.computing_intra_view_correlation(view_names[2])
        # viewDict[view_names[2]].append(self.add_feature_index)
        # gm_v3 = greedyMethod(MI_matrix, k=5, view_dict=viewDict, add_feature_index=self.add_feature_index)
        # print("将第",self.add_feature_index,"个特征加入视图",view_names[2],"后，新的视图为：")
        # print(viewDict)
        # inter12 = gm_v3.computing_inter_view_correlation(view_names[0], view_names[1])
        # inter13 = gm_v3.computing_inter_view_correlation(view_names[0], view_names[2])
        # inter23 = gm_v3.computing_inter_view_correlation(view_names[1], view_names[2])
        # viewDict[view_names[2]].remove(self.add_feature_index)
        # score[view_names[2]].append(intra3 - inter12 - inter13 - inter23)

# 利用随机数进行初始化，最好选择特征的互信息相接近
# 先是人工选择
def init_view(MI_Info, view_num):
    view_init_index = []
    feature_index = list(range(MI_Info.shape[0]))
    init_list = random.sample(feature_index, view_num)

    return view_init_index


if __name__ == '__main__':
    MI_matrix = pd.read_csv("../data/MI_info_withLabel.csv", header=None)
    # MI_matrix = MI_matrix.drop([0],axis=0)
    print(MI_matrix)
    # 初始化视图,每个视图先随机放一个特征，最好有一个初始选择条件
    # 算法对初始值敏感，初始值不好会出现特征累计现象（大部分特征加入同一个视图中）
    # 已知视图分割结果，在每个视图内随机选择一个特征加入对应视图完成初始化
    # 利用随机数进行初始化，最好选择特征的
    feature_index = list(range(MI_matrix.shape[0]))
    # init_list = random.sample(feature_index, 4)
    init_v1_index = 0
    init_v2_index = 1
    init_v3_index = 2
    # init_v4_index = 25
    # init_v5_index = 596
    # init_v6_index = 643
    viewDict = {"V1": [init_v1_index], "V2": [init_v2_index], "V3": [init_v3_index]}
                # "V5": [init_v5_index],"V6": [init_v6_index]}

    print("初始化视图为：")
    print(viewDict)

    # 将特征索引存入列表中,并去掉初始化的特征索引
    feature_index.remove(init_v1_index)
    feature_index.remove(init_v2_index)
    feature_index.remove(init_v3_index)
    #feature_index.remove(init_v4_index)
    # feature_index.remove(init_v5_index)
    # feature_index.remove(init_v6_index)

    print("---迭代开始---")
    iteration = 1
    # 启动matlab连接，在函数内启动时，每次调用都要启动一次，运行时间很长
    eng = matlab.engine.start_matlab()
    for i in feature_index:
        print("---第", iteration, "轮迭代开始---")
        # 统计每轮迭代运行时间
        time_start = time.time()
        GM = greedyMethod(MI_matrix, k=3, view_dict=viewDict, add_feature_index=i)
        viewDict = GM.feature_divide_by_greedyMethod(viewDict)
        # 打印视图构造信息
        print("第", iteration, "次迭代后，加入第", i + 1, "个特征，视图更新为：")
        print(viewDict)
        time_end = time.time()
        print("第", iteration, "次迭代运行时间", time_end - time_start)
        print("---第", iteration, "轮迭代结束---")
        if i == len(feature_index):
            # 将最终视图构造结果存为json文件
            jsObj = json.dumps(viewDict)
            view_save_path = str("../data/featureClusterResult/DVC/DVC_res_3_views_discretion_MI_coefficient.json")
            fileObject = open(view_save_path, 'w')
            fileObject.write(jsObj)
            fileObject.close()
        iteration = iteration + 1
    eng.exit()
    # # 两个指标量纲有点差距，需要标准化
    # feature_index = 3
    #
    # b = greedyMethod(MI_matrix, k=5, view_dict=viewDict, add_feature_index=feature_index)
    # view_name1 = "V1"
    # view_name2 = "V2"
    # view_name3 = "V3"
    #
    # intra1 = b.computing_intra_view_correlation(view_name1)
    # viewDict[view_name1].append(feature_index)
    # b_v1 = greedyMethod(MI_matrix, k=5, view_dict=viewDict, add_feature_index=feature_index)
    # print(viewDict)
    # inter12 = b_v1.computing_inter_view_correlation(view_name1, view_name2)
    # inter13 = b_v1.computing_inter_view_correlation(view_name1, view_name3)
    # inter23 = b_v1.computing_inter_view_correlation(view_name2, view_name3)
    # viewDict[view_name1].remove(feature_index)
    # print(inter12+inter13+inter23)
    # print(intra1, intra1 - inter12 - inter13 - inter23)
    #
    # intra2 = b.computing_intra_view_correlation(view_name2)
    # viewDict[view_name2].append(feature_index)
    # b_v2 = greedyMethod(MI_matrix, k=5, view_dict=viewDict, add_feature_index=feature_index)
    # print(viewDict)
    # inter12 = b_v2.computing_inter_view_correlation(view_name1, view_name2)
    # inter13 = b_v2.computing_inter_view_correlation(view_name1, view_name3)
    # inter23 = b_v2.computing_inter_view_correlation(view_name2, view_name3)
    # viewDict[view_name2].remove(feature_index)
    # print(inter12 + inter13 + inter23)
    # print(intra2, intra2 - inter12 - inter13 - inter23)
    #
    # intra3 = b.computing_intra_view_correlation(view_name3)
    # viewDict[view_name3].append(feature_index)
    # b_v3 = greedyMethod(MI_matrix, k=5, view_dict=viewDict, add_feature_index=feature_index)
    # print(viewDict)
    # inter12 = b_v3.computing_inter_view_correlation(view_name1, view_name2)
    # inter13 = b_v3.computing_inter_view_correlation(view_name1, view_name3)
    # inter23 = b_v3.computing_inter_view_correlation(view_name2, view_name3)
    # viewDict[view_name3].remove(feature_index)
    # print(inter12 + inter13 + inter23)
    # print(intra3, intra3 - inter12 - inter13 - inter23)