# coding: utf8
"""
homework of week2
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：“第一个数加第二个数大于第三个数加第四个数”为正样本，其余为负样本
gsh, 20230221
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import json
import matplotlib.pyplot as plt

# 定义类Net
class Net(nn.Module):
    # 定义初始化函数
    def __init__(self,input_size) -> None:
        super(Net,self).__init__()
        self.linear1 = nn.Linear(input_size, 3)
        self.activation1 = torch.sigmoid
        self.linear2 = nn.Linear(3, 1)
        self.activation2 = torch.sigmoid
        self.loss = nn.functional.mse_loss

    # 定义前向计算函数
    def forward(self, x, y=None):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        y_pred = self.activation2(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred
    # 反向传播过程不需要另外定义函数

# 单个数据点生成函数
def data_sample():
    x = np.random.random(7)
    if x[0]+x[1]>x[2]+x[3]:
        return x,1
    else:
        return x,0
    
# 数据集生成函数
def dataset_sample(sample_size):
    X = []
    Y = []
    for i in range(sample_size):
        x, y = data_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)

def evaluate(net, X_eval, Y_eval):
    net.eval()
    print('本次预测集中共有%d个正样本，%d个负样本'%(sum(Y_eval),len(Y_eval)-sum(Y_eval)))
    correct, wrong = 0, 0
    with torch.no_grad():
        Y_pred = net(X_eval)
        for Y_p,Y_t in zip(Y_pred,Y_eval):
            if float(Y_p) < 0.5 and int(Y_t) == 0:
                correct += 1
            elif float(Y_p) > 0.5 and int(Y_t) == 1:
                correct += 1
            else:
                wrong += 1
    acc = correct/(correct+wrong)
    print('正确预测个数：%d, 正确率：%f'%(correct,acc))
    return acc

def train():
    pass

def predict(saved_model, input_vec, input_size):
    net = Net(input_size)
    net.load_state_dict(torch.load(saved_model))

    net.eval()
    with torch.no_grad():
        result =net.forward(torch.FloatTensor(input_vec))
    for vec, res in zip(input_vec, result):
        print('输入：%s, 预测类别：%d, 概率值：%f'%(vec, round(float(res)),res))


def main():
    # 定义超参数
    epochs = 150 # 训练轮数
    batch_size = 10 # 每次训练样本个数
    train_sample = 150 # 每轮训练总共训练的样本总数
    evaluate_sample = 100 
    test_sample = 10 
    input_size = 7 # 输入向量维度
    lr = 0.01 # 学习率

    # 通过Net类定义具体模型对象net
    net = Net(input_size=input_size)

    # 优化器
    optim = torch.optim.Adam(net.parameters(),lr=lr)
    log = []

    # 生成数据
    X_train, Y_train = dataset_sample(train_sample)
    X_eval, Y_eval = dataset_sample(evaluate_sample)
    X_test, Y_test = dataset_sample(test_sample)

    # 训练过程
    for epoch in range(epochs):
        net.train() # 让模型处于训练模式
        all_loss = [] # 储存每次训练的loss
        for batch_index in range(train_sample//batch_size):
            X = X_train[batch_index*batch_size:(batch_index+1)*batch_size]
            Y = Y_train[batch_index*batch_size:(batch_index+1)*batch_size]
            optim.zero_grad() # 梯度归零
            loss = net(X,Y) # 同时传入X和Y，返回loss
            loss.backward() # 计算梯度
            optim.step() # 更新权重
            all_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(all_loss)))
        acc = evaluate(net, X_eval, Y_eval)
        log.append([acc, float(np.mean(all_loss))])
    
    # 保存模型
    torch.save(net.state_dict(),'saved_model.pth')

    # 画图
    print("\n\n")
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

    predict('saved_model.pth', X_test, input_size)
    return



if __name__ == '__main__':
    main()