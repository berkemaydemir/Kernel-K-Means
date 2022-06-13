from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np



# Create dataset
def make_data():
    return datasets.make_circles(n_samples=200, factor=0.5, noise=0.05)


# Dataset processing
def data_processing(data):
    data_ = list(data[0])
    data_list = []
    for d in data_:
        data_list.append(list(d))
    result = list(data[1])

    return data_list, result


# Show dataset
def show_data(data, result):
    for d in data:
        if result[data.index(d)] == 0:
            plt.scatter(d[0], d[1], c="r")
        if result[data.index(d)] == 1:
            plt.scatter(d[0], d[1], c="g")
    plt.show()


# Map raw 2D data points to 3D
# Introduce a new dimension, distance from data point to remote point, conversion: (x, y) converted to (x, sqrt(x**2+y**2), y)
def axes3d(data):
    new_data_list = []
    for d in data:
        new_data = [2*d[0]**2+2*d[1]**2, d[0]+1, d[1]+1]
        # new_data = [np.sqrt(d[0]**2+d[1]**2), d[0]+1, d[1]+1]
        new_data_list.append(new_data)

    return new_data_list


# 3D rendering
def show_data_3d(data, result):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for d in data:
        if result[data.index(d)] == 0:
            ax.scatter(d[0], d[1], d[2], c="r", marker="8")
        if result[data.index(d)] == 1:
            ax.scatter(d[0], d[1], d[2], c="g", marker="8")
    plt.show()


# 3D iterative rendering
def show_data_iter_3d(data, center):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for cls_d in data:
        for d in cls_d:
            if data.index(cls_d) == 0:
                ax.scatter(d[0], d[1], d[2], c="r", marker="8")
            if data.index(cls_d) == 1:
                ax.scatter(d[0], d[1], d[2], c="g", marker="8")

    for c in center:
        if center.index(c) == 0:
            ax.scatter(c[0], c[1], c[2], c="b", marker="v")
        if center.index(c) == 1:
            ax.scatter(c[0], c[1], c[2], c="b", marker="v")

    plt.show()


# Selection of K-mean class centers
def select_centre(data, k):
    center_list = []
    for i in range(k):
        ran = np.random.randint(len(data))
        center_list.append(data[ran])
    return center_list



def k_means(data, center):
    new_center = []
    cluster_result = []
    class_result = [[] for i in range(len(center))]
    for d in data:
        dis_list = []
        for c in center:
            # Euclidean distance for distance measurement
            dis = np.sqrt(np.sum(np.square(np.array(d)-np.array(c))))
            dis_list.append(dis)
        cluster_result.append(dis_list.index(min(dis_list)))
        class_result[dis_list.index(min(dis_list))].append(d)

    # Update Class Center
    for cls in class_result:
        x = 0
        y = 0
        z = 0
        for c in cls:
            x += c[0]
            y += c[1]
            z += c[2]
        if len(cls) == 0:
            new_center.append(center[class_result.index(cls)])
        else:
            new_center.append([round(x/len(cls), 4), round(y/len(cls), 4), round(z/len(cls), 4)])

    print(new_center)
    # print(len(class_result[0]), len(class_result[1]), len(class_result[2]))

    return cluster_result, class_result, new_center


# Reiteration
def iter_(data, center, iter):
    cluster_result, class_result, new_center = k_means(data, center)
    for i in range(iter-1):
        cluster_result, class_result, new_center = k_means(data, new_center)
        show_data_iter_3d(class_result, new_center)
    return cluster_result, class_result, new_center


if __name__ == '__main__':
    # Create dataset
    datas = make_data()
    # print(datas)
    # Data processing
    data, result = data_processing(datas)
    # print(data, result)
    # A cluster diagram showing the correct data
    show_data(data, result)
    # Perform core coordinate transformation
    data_3d = axes3d(data)
    # print(data_3d)
    # draw 3D pictures
    show_data_3d(data_3d, result)

    # select class center points for k-means
    centers = select_centre(data_3d, 2)

    cluster_results, class_results, new_centers = iter_(data_3d, centers, 10)
