import numpy as np
import pyflann
import argparse
import os
import cv2
import time
import shutil
import scipy.io as scio
import pickle
from multiprocessing import Pool
from functools import partial
# from multiprocessing.dummy import Pool
from itertools import combinations, permutations
from collections import defaultdict
from sklearn.cluster import KMeans


def get_args_from_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matfile_path', dest='matfile_path', help='用于提取feature和image的label',
                        default='/Users/qiuxiaocong/Downloads/Rank-Order-Face-Clustering/LightenedCNN_C_lfw.mat', type=str)
    parser.add_argument('--image_save_path', dest='image_save_path', help='提取出来的image的存放路径',
                        default='/Users/qiuxiaocong/Downloads/Rank-Order-Face'
                                '-Clustering/subway_imgs/', type=str)
    parser.add_argument('--root_path', dest='root_path', help='项目根目录',
                        default='/Users/qiuxiaocong/Downloads/Rank-Order-Face'
                                '-Clustering', type=str)
    parser.add_argument('--neighbor_number', dest='neighbor_number', help='最近邻列表的个数[该值不宜过大,否则会天然地使得分子过小]',
                        default=200, type=int)
    parser.add_argument('--threshold', dest='threshold', help='用于判断Rank_Order距离的阈值',
                        default=1.1, type=float)
    parser.add_argument('--cpu_num', dest='cpu_num', help='the number of cpu',
                        default=12, type=int)
    parser.add_argument('--kdtree_state', dest='kdtree_state', help='whether to use '
                                                                    'kdtree to get '
                                                                    'top-k nearest '
                                                                    'neighbors',
                        default=True, type=bool)
    args = parser.parse_args()
    return args


def get_info_from_matfile(matfile_path, root_path):
    data = scio.loadmat(matfile_path)
    labels = data['labels']           # labels
    features = data['features']       # features
    image_paths = data['image_path']  # image_paths
    # Saving Pickle
    with open(os.path.join(root_path, 'lfw_labels.pickle'), 'wb') as f:
        pickle.dump(labels, f)
    with open(os.path.join(root_path, 'lfw_features.pickle'), 'wb') as f:
        pickle.dump(features, f)
    with open(os.path.join(root_path, 'lfw_image_paths.pickle'), 'wb') as f:
        pickle.dump(image_paths, f)
    # print(data)
    # --------------------
    # print(labels.shape)
    # print(features.shape)
    # print(image_paths.shape)
    # --------------------
    # print(labels[0][0])
    # print(features[0])
    # print(image_paths[0][0])
    # print(type(data))
    # --------------------
    # return labels, features, image_paths


def get_nearest_neighbors_index(neighbor_number, kdtree_state, root_path):
    with open(os.path.join(root_path, 'lfw_features.pickle'), 'rb') as f:
        lfw_features = pickle.load(f)
    # print(lfw_features)
    # print(type(lfw_features))
    # time.sleep(1000000)
    # feature_np = np.array(lfw_features)

    pyflann.set_distance_type(distance_type='euclidean')
    flann = pyflann.FLANN()
    if kdtree_state:
        params = flann.build_index(lfw_features, algorithm='kdtree', trees=4)
        print('params -> kdtree')
    else:
        params = flann.build_index(lfw_features, algorithm="autotuned",
                                   target_precision=1.0, log_level="info")
        print('params -> autotuned')
    nearest_neighbors, distances = flann.nn_index(lfw_features, neighbor_number,
                                                  checks=params['checks'])
    # print(nearest_neighbors[0])
    # print(distances[0])
    # time.sleep(1000000)

    if kdtree_state:
        with open(os.path.join(root_path, 'lfw_nearest_neighbors_tree.pickle'), 'wb') as f:
            pickle.dump(nearest_neighbors, f)
        with open(os.path.join(root_path, 'lfw_distances_tree.pickle'), 'wb') as f:
            pickle.dump(distances, f)
        print('kdtree to get top-k nearest neighbors')
    else:
        with open(os.path.join(root_path, 'lfw_nearest_neighbors.pickle'), 'wb') as f:
            pickle.dump(nearest_neighbors, f)
        with open(os.path.join(root_path, 'lfw_distances.pickle'), 'wb') as f:
            pickle.dump(distances, f)
        print('auto algorithm to get top-k nearest neighbors')
    print('Session Done!')
    # return nearest_neighbors, distances


def make_nearest_neighbors_dict(kdtree_state, root_path):
    # 为了方便后面计算aro距离, 利用dict类型索引特征
    if kdtree_state:
        with open(os.path.join(root_path, 'lfw_nearest_neighbors_tree.pickle'), 'rb') as f:
            nearest_neighbors = pickle.load(f)
        print('Loading kdtree nearest_neighbors')
    else:
        with open(os.path.join(root_path, 'lfw_nearest_neighbors.pickle'), 'rb') as f:
            nearest_neighbors = pickle.load(f)
        print('Loading autotuned nearest_neighbors')

    nearest_neighbors_dict = {}
    for idx in range(nearest_neighbors.shape[0]):  # shape[0]对应人脸数量 shape[1]对应top-k
        nearest_neighbors_dict[idx] = nearest_neighbors[idx][:]

    if kdtree_state:
        with open(os.path.join(root_path, 'lfw_nearest_neighbors_tree_dict.pickle'),
                  'wb') as f:
            pickle.dump(nearest_neighbors_dict, f)
        print('Saving kdtree nearest_neighbors_dict')
    else:
        with open(os.path.join(root_path, 'lfw_nearest_neighbors_dict.pickle'), 'wb')\
                as f:
            pickle.dump(nearest_neighbors_dict, f)
        print('Saving autotuned nearest_neighbors_dict')
    print('Dict done.')


def make_labels_dict(root_path):
    with open(os.path.join(root_path, 'lfw_labels.pickle'), 'rb') as f:
        labels = pickle.load(f)[0]
    labels_dict = dict()

    for idx, label in enumerate(labels):
        # print("idx is:{}".format(idx))
        # print("label is:{}".format(label))
        # time.sleep(10000)
        labels_dict[idx] = label

    with open(os.path.join(root_path, 'lfw_labels_dict.pickle'), 'wb') as f:
        pickle.dump(labels_dict, f)
    print('Labels dict done.')


def calculate_approximate_rank_order_distance_for_one_row(nearest_neighbors_dict, row):
# def calculate_approximate_rank_order_distance_for_one_row(root_path, kdtree_state, nearest_neighbors_dict, row):
    """
    计算一张face与其top-k最近邻列表中所有其他人脸的aro距离
    :param row: nearest_neighbors_dict的key值
    :param root_path: 根路径
    :return: 返回[0,x,x,x,…,x]共200个元素, 首元素值为0, x表示key为row人脸与其最近邻列表中其他人
    脸的aro距离
    """
    # row是字典的key值, 从0到13232
    # 一开始的做法是依次分别计算两张人脸, 这样带来的计算量是巨大的.
    # Compute pairwise distances between each face and its top-k nearest neighbor lists
    # if kdtree_state:
    #     with open(os.path.join(root_path, 'lfw_nearest_neighbors_tree_dict.pickle'), 'rb') as f:
    #         nearest_neighbors_dict = pickle.load(f)  # 加载最近邻的列表[字典]
    #     # print('Loading kdtree nearest_neighbors_dict')
    # else:
    #     with open(os.path.join(root_path, 'lfw_nearest_neighbors_dict.pickle'), 'rb') as f:
    #         nearest_neighbors_dict = pickle.load(f)
    #     # print('Loading autotuned nearest_neighbors_dict')

    print("row is :{}".format(row))
    start = time.time()

    row_i = nearest_neighbors_dict[row]          # key值为row对应的top-k最近邻列表
    neighbor_number = len(row_i)                 # top-k中的k值
    aro_dist_row = np.zeros([1, neighbor_number])

    for idx, j in enumerate(row_i[1:]):   # row_i -> i , row_j -> j
        Oi_j = idx + 1      # Oi(j) 人脸j在人脸i的最近邻列表的索引位置
        i_in_j = True       # 人脸i是否在人脸j的最近邻列表中
        try:
            row_j = nearest_neighbors_dict[j]
            Oj_i = np.where(row_j == row)[0][0] + 1   # 列索引+1
        except IndexError:
            Oj_i = neighbor_number + 10  # here
            # i_in_j = False
            aro_dist_row[0, Oi_j] = 99999.0
            continue

        face_i_1 = set(row_i[:Oi_j])
        face_j_1 = set(nearest_neighbors_dict[j])
        d_i_j = len(face_i_1.difference(face_j_1))

        face_i_2 = set(row_i)
        face_j_2 = set(nearest_neighbors_dict[j][:Oj_i])
        d_j_i = len(face_j_2.difference(face_i_2))

        # if not i_in_j:
        #     aro_dist_row[0, Oi_j] = 99999.0
        # else:
        aro_dist_row[0, Oi_j] = float(d_i_j + d_j_i)/min(Oi_j, Oj_i)

    # result = dict()               # map()得到的结果是顺序的, 可以通过添加key值的方式验证
    # result[row] = aro_dist_row    # map()得到的结果是顺序的, 可以通过添加key值的方式验证
    # return result
    print('take {}'.format(time.time()-start))
    return aro_dist_row


def calculate_approximate_rank_order_distance_for_all(nearest_neighbors, cpu_num, kdtree_state, root_path, nearest_neighbors_dict):
    start = time.time()
    # if kdtree_state:
    #     with open(os.path.join(root_path, 'lfw_nearest_neighbors_tree.pickle'), 'rb') as f:
    #         nearest_neighbors = pickle.load(f)
    #     print('Loading kdtree nearest_neighbors')
    # else:
    #     with open(os.path.join(root_path, 'lfw_nearest_neighbors.pickle'), 'rb') as f:
    #         nearest_neighbors = pickle.load(f)
    #     print('Loading autotuned nearest_neighbors')

    face_num = nearest_neighbors.shape[0]
    roc = np.zeros(nearest_neighbors.shape)      # 最后的结果

    pool = Pool(processes=cpu_num)
    func = partial(calculate_approximate_rank_order_distance_for_one_row, nearest_neighbors_dict)
    result = pool.map(func, range(face_num))   # map方法用于多个任务同步
    pool.close()
    # result 形式为:[[], [], [], [], …, []] ?
    for idx, value in enumerate(result):
        roc[idx, :] = value

    end = time.time()
    print("Distance calculation time : {}".format(end - start))

    if kdtree_state:
        with open(os.path.join(root_path, 'lfw_aro_dist_tree_jc.pickle'), 'wb') as f:
            pickle.dump(roc, f)
        print('Saving kdtree aro_dist')
    else:
        with open(os.path.join(root_path, 'lfw_aro_dist_jc.pickle'), 'wb') as f:
            pickle.dump(roc, f)
        print('Saving autotuned aro_dist')
    print('Aro done.')


def merge_nearest_neighbors_below_threshold(kdtree_state, threshold, root_path):
    if kdtree_state:
        with open(os.path.join(root_path, 'lfw_aro_dist_tree_jc.pickle'), 'rb') as f:
            aro_dist = pickle.load(f)
        print('Loading tree aro_dist')

        with open(os.path.join(root_path, 'lfw_nearest_neighbors_tree_dict.pickle'), 'rb') as f:
            nearest_neighbors_dict = pickle.load(f)
        print('Loading tree nearest_neighbors_dict')
    else:
        with open(os.path.join(root_path, 'lfw_aro_dist_jc.pickle'), 'rb') as f:
            aro_dist = pickle.load(f)
        print('Loading aotutuned aro_dist')
        with open(os.path.join(root_path, 'lfw_nearest_neighbors_dict.pickle'), 'rb') as f:
            nearest_neighbors_dict = pickle.load(f)
        print('Loading autotuned nearest_neighbors_dict')

    # dictionary
    plausible_nearest_neighbors = dict()
    for idx in range(len(nearest_neighbors_dict)):
        plausible_nearest_neighbors[idx] = set(list(np.take(nearest_neighbors_dict[idx], np.where(aro_dist[idx] <= threshold)[0])))

    if kdtree_state:
        with open(os.path.join(root_path, 'lfw_plausible_nearest_neighbors_tree.pickle'), 'wb') as f:
            pickle.dump(plausible_nearest_neighbors, f)
        print('Saving kdtree plausible_nearest_neighbors')
    else:
        with open(os.path.join(root_path, 'lfw_plausible_nearest_neighbors.pickle'), 'wb') as f:
            pickle.dump(plausible_nearest_neighbors, f)
        print('Saving autotuned plausible_nearest_neighbors')

    print('plausible done')


def transitively_merge_all_pairs_of_faces(kdtree_state, root_path):
    if kdtree_state:
        with open(os.path.join(root_path, 'lfw_plausible_nearest_neighbors_tree.pickle'), 'rb') as f:
            plausible_nearest_neighbors = pickle.load(f)
        print('Loading tree plausible_nearest_neighbors')
    else:
        with open(os.path.join(root_path, 'lfw_plausible_nearest_neighbors.pickle'), 'rb') as f:
            plausible_nearest_neighbors = pickle.load(f)
        print('Loading aotutuned plausible_nearest_neighbors')

    face_ids = set(list(np.arange(0, len(plausible_nearest_neighbors))))
    # 因为等会需要用到集合的操作
    clusters = []

    while face_ids:
        face_id = face_ids.pop()
        group = {face_id}
        queue = [face_id]  # 存放属于一个类需要归并簇类的所有对象
        while queue:
            face_id = queue.pop(0)
            p_nearest_neighbors = plausible_nearest_neighbors[face_id]
            exist_face = face_ids.intersection(p_nearest_neighbors)  # 取交集
            # 已经不在face_ids中的就不用再做无效查找了 要确保需要扩散的点仍然在face_ids中
            exist_face.difference_update(group)  # 需要把自身去掉 如27:{27,28,29} 去掉27

            face_ids.difference_update(exist_face)
            group.update(exist_face)
            queue.extend(list(exist_face))
        clusters.append(group)

    if kdtree_state:
        with open(os.path.join(root_path, 'lfw_clusters_tree.pickle'), 'wb') as f:
            pickle.dump(clusters, f)
        print('Saving kdtree clusters result')
    else:
        with open(os.path.join(root_path, 'lfw_clusters.pickle'), 'wb') as f:
            pickle.dump(clusters, f)
        print('Saving autotuned clusters result')

    print('merge done')


def counting_pairs(labels_dict, cluster):
    # input:一个簇类, 计算TP
    # time1 = time.time()
    # with open(os.path.join(root_path, 'lfw_labels_dict.pickle'), 'rb') as f:
    #     labels_dict = pickle.load(f)
    # print('labels_dict load take :{}s'.format(time.time()-time1))
    correct_pairs = 0
    total_pairs = 0

    pairs = combinations(list(cluster), 2)

    for face_id1, face_id2 in pairs:
        if labels_dict[face_id1] == labels_dict[face_id2]:
            correct_pairs += 1
        total_pairs += 1
    return correct_pairs, total_pairs


def clustering_evaluation(kdtree_state, root_path):
    time1 = time.time()
    if kdtree_state:
        with open(os.path.join(root_path, 'lfw_clusters_tree.pickle'), 'rb') as f:
            clusters = pickle.load(f)
        print('Loading tree clusters result')
    else:
        with open(os.path.join(root_path, 'lfw_clusters.pickle'), 'rb') as f:
            clusters = pickle.load(f)     # 聚类结果
        print('Loading aotutuned clusters result')

    clusters_num = len(clusters)

    with open(os.path.join(root_path, 'lfw_labels_dict.pickle'), 'rb') as f:
        labels_dict = pickle.load(f)

    time2 = time.time()
    print('labels_dict_pikcle take :{}'.format(time2-time1))

    correct_pairs = 0     # 对应TP
    total_pairs = 0       # 对应TP+FP(预测为真)
    act_pairs = 0         # 对应TP+FN 真正的类别(个数大于2) 实际为真 由 labels_dict 计算得到

    for cluster in clusters:
        correct_pair, total_pair = counting_pairs(labels_dict, cluster)
        correct_pairs = correct_pairs + correct_pair
        total_pairs = total_pairs + total_pair

    new_cluster = defaultdict(list)
    # defaultdict 作用是在于当字典里的key不存在但被查找时, 返回的不是keyError而是一个默认值
    # 这个默认值由括号后定义的类型决定:list对应[], str对应"", int对应0, set对应()
    for face_id, label in labels_dict.items():
        new_cluster[label].append(face_id)

    for label_id, all_sub_face_id in new_cluster.items():
        sub_cluster_num = len(all_sub_face_id)
        act_pairs = act_pairs + (sub_cluster_num * (sub_cluster_num - 1))/2.0

    # print('correct pair :{}'.format(correct_pairs))
    # print('total pair :{}'.format(total_pairs))
    # print('act pair :{}'.format(act_pairs))

    precision = float(correct_pairs)/total_pairs
    recall = float(correct_pairs)/act_pairs
    f1_score = (2 * precision * recall) / (precision + recall) \
        if precision + recall > 0 else 0

    print('Clusters number:{}'.format(clusters_num))
    print('Precision:{}'.format(precision))
    print('Recall:{}'.format(recall))
    print('F1 Score:{}'.format(f1_score))


def k_means_test():
    with open(os.path.join('/Users/qiuxiaocong/Downloads/Rank-Order-Face-Clustering',
                           'lfw_features.pickle'), 'rb') as f:
        features = pickle.load(f)
    print(type(features))
    print(features.shape)
    start = time.time()
    kmeans = KMeans(n_clusters=100, random_state=0).fit(features)
    print(type(kmeans.labels_))
    print('kmeans take :{}s'.format(time.time()-start))

    with open(os.path.join('/Users/qiuxiaocong/Downloads/Rank-Order-Face-Clustering',
                           'lfw_labels_dict.pickle'), 'rb') as f:
        labels_dict = pickle.load(f)

    kmeans_labels = list(kmeans.labels_)
    # print(kmeans_labels)
    # print(len(kmeans_labels))

    label_dict = dict()
    for idx, label in enumerate(kmeans_labels):
        label_dict[idx] = label

    new_label = defaultdict(list)
    for key, value in label_dict.items():
        new_label[value].append(key)

    clusters = []
    for key, value in new_label.items():
        clusters.append(set(value))
    # print(clusters)
    print(len(clusters))

    correct_pairs = 0     # 对应TP
    total_pairs = 0       # 对应TP+FP(预测为真)
    act_pairs = 0         # 对应TP+FN 真正的类别(个数大于2) 实际为真 由 labels_dict 计算得到

    for cluster in clusters:
        correct_pair, total_pair = counting_pairs(labels_dict, cluster)
        correct_pairs = correct_pairs + correct_pair
        total_pairs = total_pairs + total_pair

    new_cluster = defaultdict(list)
    # defaultdict 作用是在于当字典里的key不存在但被查找时, 返回的不是keyError而是一个默认值
    # 这个默认值由括号后定义的类型决定:list对应[], str对应"", int对应0, set对应()
    for face_id, label in labels_dict.items():
        new_cluster[label].append(face_id)

    for label_id, all_sub_face_id in new_cluster.items():
        sub_cluster_num = len(all_sub_face_id)
        act_pairs = act_pairs + (sub_cluster_num * (sub_cluster_num - 1))/2.0

    precision = float(correct_pairs)/total_pairs
    recall = float(correct_pairs)/act_pairs
    f1_score = (2 * precision * recall) / (precision + recall) \
        if precision + recall > 0 else 0

    print('Clusters number:{}'.format(100))
    print('Precision:{}'.format(precision))
    print('Recall:{}'.format(recall))
    print('F1 Score:{}'.format(f1_score))


def main():
    start = time.time()
    args = get_args_from_command_line()
    # get_info_from_matfile(args.matfile_path, args.root_path)
    get_nearest_neighbors_index(args.neighbor_number, args.kdtree_state, args.root_path)
    time1 = time.time()
    make_nearest_neighbors_dict(args.kdtree_state, args.root_path)
    make_labels_dict(args.root_path)
    time2 = time.time()
    if args.kdtree_state:
        with open(os.path.join(args.root_path, 'lfw_nearest_neighbors_tree_dict.pickle'), 'rb') as f:
            nearest_neighbors_dict = pickle.load(f)  # 加载最近邻的列表[字典]
    else:
        with open(os.path.join(args.root_path, 'lfw_nearest_neighbors_dict.pickle'), 'rb') as f:
            nearest_neighbors_dict = pickle.load(f)
    time3 = time.time()
    #
    if args.kdtree_state:
        with open(os.path.join(args.root_path, 'lfw_nearest_neighbors_tree.pickle'), 'rb') as f:
            nearest_neighbors = pickle.load(f)
    else:
        with open(os.path.join(args.root_path, 'lfw_nearest_neighbors.pickle'), 'rb') as f:
            nearest_neighbors = pickle.load(f)
    calculate_approximate_rank_order_distance_for_all(nearest_neighbors, args.cpu_num, args.kdtree_state, args.root_path, nearest_neighbors_dict)
    time4 = time.time()
    merge_nearest_neighbors_below_threshold(args.kdtree_state, args.threshold, args.root_path)
    time5 = time.time()
    transitively_merge_all_pairs_of_faces(args.kdtree_state, args.root_path)
    time6 = time.time()
    clustering_evaluation(args.kdtree_state, args.root_path)
    time7 = time.time()
    #
    print('time1-start:{}'.format(time1 - start))
    print('time2-time1:{}'.format(time2 - time1))
    print('time3-time2:{}'.format(time3 - time2))
    print('time4-time3:{}'.format(time4 - time3))
    print('time5-time4:{}'.format(time5 - time4))
    print('time6-time5:{}'.format(time6 - time5))
    print('time7-time6:{}'.format(time7 - time6))

    # print('Time taken for clustering: {:.3f} seconds'.format(time.time()-start))
    # print('eva take :{}'.format(time.time()-mid))

    # k_means_test()


if __name__ == '__main__':
    main()




