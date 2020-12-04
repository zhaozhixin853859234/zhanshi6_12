import pandas as pd
from sklearn import preprocessing


def norm(data):
    columns_names = data.columns
    # print(df_data.shape)
    # print(df_label.shape)
    min_max_sacler = preprocessing.MinMaxScaler()
    min_max_sacler.fit(data)
    temp = min_max_sacler.transform(data)
    res = pd.DataFrame(temp, columns=columns_names)
    return res

if __name__ == '__main__':
    df_test = pd.read_csv("../data/1791269791_test_grade.csv", header=0)
    df_train = pd.read_csv("../data/1791269877_train.csv", header=0)

    df_test_feature = df_test.drop(["date","maxValue_max","grade"],axis=1)
    df_test_label = df_test[["maxValue_max", "grade"]]

    df_train_feature = df_train.drop(["date","maxValue_max"],axis=1)
    df_train_label = df_train["maxValue_max"]

    df_test_feature_norm = norm(df_test_feature)
    df_train_feature_norm = norm(df_train_feature)

    norm_test = pd.concat([df_test_feature_norm,pd.DataFrame(df_test_label,columns=["maxValue_max", "grade"])],axis=1)
    norm_test = pd.DataFrame(norm_test)
    norm_train = pd.concat([df_train_feature_norm, pd.DataFrame(df_train_label,columns=["maxValue_max"])],axis=1)
    norm_train = pd.DataFrame(norm_train)

    norm_train.to_csv("../data/1791269877_train_norm.csv",index=None)
    norm_test.to_csv("../data/1791269791_test_grade_norm.csv",index=None)
    # print(norm_train)
    # print(norm_test)