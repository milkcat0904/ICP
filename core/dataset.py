import numpy as np
from tqdm import tqdm

class MakePointCloudData():
    def __init__(self, cfg):
        self.cfg = cfg['data']['synthetic']
        self.num_points = self.cfg['num_points']
        self.num_test = self.cfg['num_test']
        self.data_dim = self.cfg['data_dim']
        self.origin_data = np.random.rand(self.num_points, self.data_dim)
        self.target_data = np.zeros((self.num_test, self.num_points, self.data_dim))
        self.shuffle = self.cfg['shuffle']

    def make_target_data(self):
        print('Start make {} target data...'.format(self.num_test))

        for i in tqdm(range(self.num_test)):
            tmp_target = np.copy(self.origin_data)

            # translate & rotation & noise
            t = np.random.rand(self.data_dim)* self.cfg['translation_scale']
            tmp_target += t
            R_mat = self.rotation_matrix(np.random.rand(self.data_dim), np.random.rand()*self.cfg['rotation_scale'])
            tmp_target = np.dot(R_mat, tmp_target.T).T
            tmp_target += np.random.randn(self.num_points, self.data_dim)* self.cfg['noise_sigma']

            # registration
            if self.shuffle:
                np.random.shuffle(tmp_target)

            self.target_data[i] = tmp_target

        return self.origin_data, self.target_data

    def rotation_matrix(self, axis, theta):
        axis = axis/np.sqrt(np.dot(axis, axis))
        a = np.cos(theta/2.)
        b, c, d = -axis*np.sin(theta/2.)

        return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                         [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                         [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])