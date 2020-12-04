import pandas as pd
if __name__ == '__main__':
    df = pd.read_csv("../data/1791269877_train.csv",header=0)
    # 将浓度按阈值分级，[0,0.5)为1级，[0.5.0.75)为2级，[0.75,1)为3级，[1,99)为4级
    grade_list = []
    maxValue_max = list(df["maxValue_max"])
    for i in maxValue_max:
        if i>=1:
            grade_list.append(4)
        elif i<1 and i>=0.7:
            grade_list.append(3)
        elif i<0.7 and i>=0.4:
            grade_list.append(2)
        else:
            grade_list.append(1)
    grade = pd.DataFrame(grade_list, columns=["grade"])
    data = pd.concat([df, grade], axis=1)
    data = pd.DataFrame(data)
    data.to_csv("../data/1791269877_train_grade.csv",index=None)
    print(data["grade"].value_counts())
    # print(data["grade"].value_counts())
    # print(maxValue_max)
    # print(grade_list)
    # print(len(grade_list))
    # print(len(maxValue_max))
    # print(df.shape)