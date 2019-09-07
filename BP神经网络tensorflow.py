# -*- coding: utf-8 -*- 
# @Author : yunze


import pandas as pd
import tensorflow as tf
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
data_train = pd.read_csv('iris_training.csv', names=CSV_COLUMN_NAMES, header=0)
data_test = pd.read_csv('iris_test.csv', names=CSV_COLUMN_NAMES, header=0)


# 将属性值和标记值分开
train_x, train_y = data_train, data_train.pop('Species')
test_x, test_y = data_test, data_test.pop('Species')


# 定义存储特征的列表，存贮特征值
my_feature_columns = []
# 获取训练集的特征
for key in train_x.keys():
    print(key)
    # feature_column.numeric_column表示tensorflow的特征工程
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# 建训练模型
classifier = tf.estimator.DNNClassifier(
    # 这个模型接收哪些输入的特征
    feature_columns=my_feature_columns,
    # 包含两个隐含层，两个隐含层包含10个神经元
    hidden_units=[10, 10],
    # 最终结果要分成几类
    n_classes=3
)

# 训练模型

# 为模型提供数据并进行训练,由于tensorflow采用 一个批量梯度下降算法更新参数，
# 这里可以构造一个函数来生成数据，并且可以在这个函数当中对数据进行打乱。


def train_func(train_x, train_y):
    # 使用data.Dataset.from_tensor_slices将数据传递给TensorFlow，从数组中提取切片
    dataset = tf.data.Dataset.from_tensor_slices((dict(train_x), train_y))
    print(dataset)
    # shuffle用来打乱，数值越大，混乱程度就越大，batch表示一批，一般默认repeat()为空
    dataset = dataset.shuffle(1000).repeat().batch(100)
    return dataset

# 进行模型训练，进行1000 个回合的训练，每次100调数据


classifier.train(
    input_fn=lambda: train_func(train_x, train_y),
    steps=1000)

# 模型预测


def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    return dataset

# 预测模型


predict_arr = []
predictions = classifier.predict(
        input_fn=lambda: eval_input_fn(test_x, labels=test_y, batch_size=100))
print(predictions)
for predict in predictions:
    print(predict)
    predict_arr.append(predict['probabilities'].argmax())
# = 代表赋值，== 代表是是否相等，返回值是True or false
result = predict_arr == test_y
print(result)
# 遍历比较是否相等的返回结果，将比较为True的结果返回到result1列表中
result1 = [w for w in result if w is True]
print("准确率为 %s" % str((len(result1)/len(result))))


