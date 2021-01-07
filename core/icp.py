'''
    ICP:
        1. 归一化得到p'、q'，均值为质心mu_a, mu_b
        2. H = p'.T * q'
        3. SVD H = U * S * Vt
        4. R = Vt.T * U.T
        5. t = mu_b.T - (R*mu_a.T)
'''

import os
import numpy as np
from tqdm import trange
from sklearn.neighbors import NearestNeighbors

class ICP():
    def __init__(self, origin, target, cfg):
        self.cfg = cfg
        self.origin = origin
        self.target = target
        self.allow_error = self.cfg['icp']['allow_error']
        self.target_num = target.shape[0]
        self.output_folder = 'output/'

    def fit_transform(self, points_A, points_B):
        '''

        :param points_A: shape (num_points, data_dim)
        :param points_B: shape (num_points, data_dim)
        :return: T, R, t
        '''
        assert points_A.shape == points_B.shape
        data_dim = points_B.shape[1]

        # 计算质心, p', q'
        mu_A = np.mean(points_A, axis = 0)
        mu_B = np.mean(points_B, axis = 0)
        points_A = points_A - mu_A
        points_B = points_B - mu_B

        # SVD求解Rotation matrix
        H = np.dot(points_A.T, points_B)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # det(R)如果是负的时候，Vt的第2行乘以-1
        if np.linalg.det(R) < 0:
            Vt[data_dim-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        # 根据R计算t
        t = mu_B.T - np.dot(R, mu_A.T)

        # 变换矩阵T = （R，t）
        T = np.identity(data_dim + 1) # 4x4的单位矩阵
        T[:data_dim, :data_dim] = R
        T[:data_dim, data_dim] = t
        return T, R, t

    def run_icp(self):
        print('Start ICP...')

        for i in trange(self.target_num):
            tmp_point_B = self.target[i]

            T, _, _ = self.fit_transform(self.origin, tmp_point_B)

            # 对origin扩充维度，使得可以和4x4的卷积相乘 (num_points, 3) -> (num_points, 4)
            tmp_extend = np.ones((tmp_point_B.shape[0], tmp_point_B.shape[1]+1))
            tmp_extend[:, :tmp_point_B.shape[1]] = self.origin

            out_B = np.dot(T, tmp_extend.T).T[:, :tmp_point_B.shape[1]]
            # assert np.allclose(out_B, tmp_point_B, atol = 6 * self.allow_error)

            error = np.sum((out_B - tmp_point_B)**2)/out_B.shape[0]
            print('Mean Square Error: {}'.format(error))

            # 保存第一组数据用于meshlab可视化
            if i == 0:
                self.write_off_file(self.origin, 'orgin.off')
                self.write_off_file(tmp_point_B, 'target.off')
                self.write_off_file(out_B, 'icp_out.off')

        print('ICP finished.')

    def run_registration_icp(self):
        print('Start Point Cloud Registration + ICP...')
        for i in trange(self.target_num):
            tmp_point_B = self.target[i]

            prev_error = 0
            indices = []

            # registration
            tmp_A = self.origin
            for j in range(self.cfg['registration']['num_iteration']):

                # 从B里找A的最近邻点，A里每个点与最近邻点的距离
                distances, indices = self.nearest_neighbor(tmp_A, tmp_point_B)

                # 按照index调整B'，A与B'进行fit
                T, _, _ = self.fit_transform(tmp_A, tmp_point_B[indices, :])
                tmp_extend = np.ones((tmp_A.shape[0], tmp_A.shape[1] + 1))
                tmp_extend[:, :tmp_A.shape[1]] = tmp_A

                # 将A进行变换
                tmp_A = np.dot(T, tmp_extend.T).T[:, :tmp_A.shape[1]]

                # distance作为误差
                mean_error = np.mean(distances)
                if np.abs(prev_error - mean_error) < self.cfg['registration']['tolerance']:
                    break
                prev_error = mean_error

            # icp
            tmp_point_B = tmp_point_B[indices, :]
            T, _, _ = self.fit_transform(self.origin, tmp_point_B)

            # 对origin扩充维度，使得可以和4x4的卷积相乘 (num_points, 3) -> (num_points, 4)
            tmp_extend = np.ones((tmp_point_B.shape[0], tmp_point_B.shape[1]+1))
            tmp_extend[:, :tmp_point_B.shape[1]] = self.origin

            out_B = np.dot(T, tmp_extend.T).T[:, :tmp_point_B.shape[1]]
            assert np.allclose(out_B, tmp_point_B, atol = 6 * self.allow_error)

            error = np.sum((out_B - tmp_point_B)**2)/out_B.shape[0]
            print('Mean Square Error: {}'.format(round(error, 6)))

            # 保存第一组数据用于meshlab可视化
            if i == 0:
                self.write_off_file(self.origin, 'orgin.off')
                self.write_off_file(tmp_point_B, 'target.off')
                self.write_off_file(out_B, 'icp_out.off')
        print('ICP finished')

    def nearest_neighbor(self, src, dst):
        '''
        Find the nearest (Euclidean) neighbor in dst for each point in src

        从dst点里找到对于src的每个点的最近邻

        Input:
            src: Nxm array of points
            dst: Nxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor 每个src的最近邻点之间的距离
            indices: dst indices of the nearest neighbor 最近邻点dst的index
        '''

        assert src.shape == dst.shape

        neigh = NearestNeighbors(n_neighbors = 1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance = True)
        return distances.ravel(), indices.ravel()

    def write_off_file(self, points, file_name):
        output_path = os.path.join(self.output_folder, file_name)
        out = ['OFF\n', '{} 0 0\n'.format(points.shape[0])]

        for i in range(points.shape[0]):
            tmp = ' '.join(list(map(str, points[i].tolist()))) + '\n'
            out.append(tmp)

        with open(output_path, 'w') as w:
            w.writelines(out)




