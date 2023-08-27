from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 组合所有数据和标签
data = np.concatenate((data1, data2, data3, data4, data5), axis=0)
labels = np.concatenate((label1, label2, label3, label4, label5), axis=0)

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=0)
data_2d = tsne.fit_transform(data)

# 创建一个新的图形
plt.figure(figsize=(6, 6))

# 根据你的标签数量和数据，这里可能需要做一些修改
# 在这个例子中，我假设标签是一个整数列表，范围从1到5
colors = 'r', 'g', 'b', 'c', 'm'
for i, color in zip(range(1, 6), colors):
    plt.scatter(data_2d[labels == i, 0], data_2d[labels == i, 1], c=color, label=str(i))

plt.legend()
plt.show()
