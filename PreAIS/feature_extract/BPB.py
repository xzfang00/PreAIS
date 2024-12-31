import numpy as np


def bpb_extract_features(sequences, labels):

    AA = 'AUGC'
    if labels != None:

        pos_sequences = [seq for seq, label in zip(sequences, labels) if label == 1]  # 提取正样本
        neg_sequences = [seq for seq, label in zip(sequences, labels) if label == 0]  # 提取负样本

        n1 = len(pos_sequences)
        n2 = len(neg_sequences)

        if n1 == 0 or n2 == 0:
            return np.array([])

        M1 = len(pos_sequences[0])  # 正样本序列的长度
        M2 = len(neg_sequences[0])  # 负样本序列的长度

        # 记录每两个核苷酸在每个位置出现的次数
        F1 = np.zeros((4, M1))  # 用于正样本
        F2 = np.zeros((4, M2))  # 用于负样本

        # 统计正样本中的核苷酸对
        for seq in pos_sequences:
            for i in range(M1):
                s = seq[i]
                i1 = AA.index(s)
                F1[i1, i] += 1

        # 统计合并样本中的核苷酸对
        for seq in neg_sequences:
            for i in range(M2):
                s = seq[i]
                i1 = AA.index(s)
                F2[i1, i] += 1

        # 归一化
        F1 /= n1
        F2 /= n2
    else:
        F1 = np.load('./feature_extract/F1.npy')
        F2 = np.load('./feature_extract/F2.npy')
    # 构建特征向量
    feature_vectors = np.zeros((len(sequences), 2*len(sequences[0])))

    for m in range(len(sequences)):
        for k in range(len(sequences[0])):
            s = sequences[m][k]
            i = AA.index(s)

            feature_vectors[m, k] = F1[i, k]
            feature_vectors[m, len(sequences[0]) + k] = F2[i, k]

    return feature_vectors

"""
# 示例使用
sequences = ['AUGC', 'UGCA', 'GCAU']  # 示例序列
labels = [1, 0, 1]  # 示例标签
features = bpb_extract_features(sequences, labels)
print(features)
"""



