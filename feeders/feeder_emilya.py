import numpy as np
from copy import deepcopy
import pickle

from torch.utils.data import Dataset

from feeders import tools
import torch
import torch.nn as nn


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, angle=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.angle = angle
        # cosine
        self.cos = nn.CosineSimilarity(dim=0, eps=0)
        self.load_data()
        if normalization:
            self.get_mean_map()

    ## todo: 修改
    def load_data(self):
        # data: N C V T M
        if self.split == 'train':
            train_data = np.load(self.data_path)
            # 取上半身的关节点，索引为0-14和23:25
            self.data = np.concatenate((train_data[:, :, :, :15, :], train_data[:, :, :, 23:26, :]), axis=3)
            with open(self.label_path, "rb") as f:
                self.label = pickle.load(f)
        elif self.split == 'eval':
            eval_data = np.load(self.data_path)
            # 取上半身的关节点，索引为0-14和23:25
            self.data = np.concatenate((eval_data[:, :, :, :15, :], eval_data[:, :, :, 23:26, :]), axis=3)
            with open(self.label_path, "rb") as f:
                self.label = pickle.load(f)
                self.sample_name = ['eval_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            test_data = np.load(self.data_path)
            # 取上半身的关节点，索引为0-14和23:25
            self.data = np.concatenate((test_data[:, :, :, :15, :], test_data[:, :, :, 23:26, :]), axis=3)
            with open(self.label_path, "rb") as f:
                self.label = pickle.load(f)
                self.sample_name = ['test_' + str(i) for i in range(len(self.data))]

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs import emilya_pairs_upper ## todo: 修改
            bone_data_numpy = np.zeros_like(data_numpy) # 3, T, V
            for v1, v2 in emilya_pairs_upper:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.angle:
            from .bone_pairs import aide_bone_angle_pairs_upper_head ## todo: 修改
            angle_data = torch.from_numpy(data_numpy) # 3, T, V
            fp_sp_joint_list_bone_angle = []
            fp_sp_two_wrist_angle = []
            fp_sp_two_elbow_angle = []
            all_angle = [fp_sp_joint_list_bone_angle, fp_sp_two_wrist_angle, fp_sp_two_elbow_angle]
            for a_key in aide_bone_angle_pairs_upper_head.keys(): ## todo: 修改
                # bone angles
                v1, v2 = aide_bone_angle_pairs_upper_head[a_key] ## todo: 修改
                vec1 = angle_data[:, :, v1 - 1] - angle_data[:, :, a_key - 1]
                vec2 = angle_data[:, :, v2 - 1] - angle_data[:, :, a_key - 1]
                angular_feature = (1.0 - self.cos(vec1, vec2))
                # 将NaN值转化为0
                angular_feature[angular_feature != angular_feature] = 0
                fp_sp_joint_list_bone_angle.append(angular_feature)
            
                # two wrist angle
                vec1 = angle_data[:, :, 11 - 1] - angle_data[:, :, a_key - 1]
                vec2 = angle_data[:, :, 10 - 1] - angle_data[:, :, a_key - 1]
                angular_feature = (1.0 - self.cos(vec1, vec2))
                angular_feature[angular_feature != angular_feature] = 0
                fp_sp_two_wrist_angle.append(angular_feature)

                # two elbow angle
                vec1 = angle_data[:, :, 9 - 1] - angle_data[:, :, a_key - 1]
                vec2 = angle_data[:, :, 8 - 1] - angle_data[:, :, a_key - 1]
                angular_feature = (1.0 - self.cos(vec1, vec2))
                angular_feature[angular_feature != angular_feature] = 0
                fp_sp_two_elbow_angle.append(angular_feature)

            for a_list_id in range(len(all_angle)):
                all_angle[a_list_id] = torch.stack(all_angle[a_list_id]) # [V, T, M]
                all_angle[a_list_id] = all_angle[a_list_id].squeeze(2)
                all_angle[a_list_id] = all_angle[a_list_id].transpose(0, 1) # [T, V]
                # print(all_angle[a_list_id].shape)
            data_numpy = torch.stack(all_angle)
            data_numpy = data_numpy.unsqueeze(-1)
            # print(data_numpy.shape)
        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
