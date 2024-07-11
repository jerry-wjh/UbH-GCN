import argparse
import pickle
import os

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import math
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn import manifold
from visdom import Visdom

EMOTION_LABEL= ['Anxiety', 'Peace', 'Weariness', 'Happiness', 'Anger']

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def show_important_joints(result):
        first_sum = np.sum(result[:,:,0], axis=0)
        first_index = np.argsort(-first_sum) + 1
        print('Weights of all joints:')
        print(first_sum)
        print('')
        print('Most important joints:')
        print(first_index)
        print('')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'AIDE'},
                        help='the work folder for storing results')
    parser.add_argument('--main-dir',
                        help='')
    parser.add_argument('--root-1',
                        type=str2bool,
                        default=True)
    parser.add_argument('--root-14',
                        type=str2bool,
                        default=True)
    parser.add_argument('--best-epoch',
                        type=int,
                        help='')
    
    arg = parser.parse_args()

    dataset = arg.dataset
    if 'AIDE' in arg.dataset:
        label = []
        with open('/home/wujiehui/dataset/AIDE/emotion_dataset_upper_head/test_label.pkl', 'rb') as f:
            label = pickle.load(f)
    else:
        raise NotImplementedError
    
    dir_cnt = 0

    if arg.CoM_1:
        with open(os.path.join(arg.main_dir, 'joint_root_1/', 'epoch1_test_score.pkl'), 'rb') as r1:
            r1 = list(pickle.load(r1).items())
        with open(os.path.join(arg.main_dir, 'bone_root_1/', 'epoch1_test_score.pkl'), 'rb') as r2:
            r2 = list(pickle.load(r2).items())
        dir_cnt += 2
        
    if arg.CoM_14:
        with open(os.path.join(arg.main_dir, 'joint_root_14/' 'epoch1_test_score.pkl'), 'rb') as r3:
            r3 = list(pickle.load(r3).items())
        with open(os.path.join(arg.main_dir, 'bone_root_14/', 'epoch1_test_score.pkl'), 'rb') as r4:
            r4 = list(pickle.load(r4).items())
        dir_cnt += 2

    right_num = total_num = right_num_5 = 0

    norm = lambda x: x / np.linalg.norm(x)

    if dir_cnt == 4:
        r = None
        TP, FP, TN, FN = 0, 0, 0, 0
        y_true = []
        y_pred = []
        out_put = []
        for i in tqdm(range(len(label))):
            l = label[i]
            if arg.CoM_1:
                r11 = np.array(r1[i][1])
                r22 = np.array(r2[i][1])
                r = norm(r11) + norm(r22)
            if arg.CoM_14:
                r33 = np.array(r3[i][1])
                r44 = np.array(r4[i][1])
                r = r + norm(r33) + norm(r44) if r is not None else norm(r33) + norm(r44)

            out_put.append(r)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1

            # Calculate F1 score
            y_true.append(int(l))
            y_pred.append(r)

        
        acc = right_num / total_num
        print('Acc: {:.4f}%'.format(acc * 100))

        out_put = np.array(out_put)
        labels = np.array(y_true)
        F1 = f1_score(y_true, y_pred, average='weighted')
        print('Weighted F1: {:.4f}%'.format(F1 * 100))

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix:')
        print(cm)

    elif dir_cnt == 2:
        r = None
        TP, FP, TN, FN = 0, 0, 0, 0
        y_true = []
        y_pred = []
        out_put = []
        for i in tqdm(range(len(label))):
            l = label[i]
            if arg.CoM_1:
                r11 = np.array(r1[i][1])
                r22 = np.array(r2[i][1])
                r = norm(r11) + norm(r22)       
            if arg.CoM_14:
                r33 = np.array(r3[i][1])
                r44 = np.array(r4[i][1])
                r = r + norm(r33) + norm(r44) if r is not None else norm(r33) + norm(r44)

            out_put.append(r)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1

            # Calculate F1 score
            y_true.append(int(l))
            y_pred.append(r)
        
        acc = right_num / total_num
        print('Acc: {:.4f}%'.format(acc * 100))

        F1 = f1_score(y_true, y_pred, average='weighted')
        print('Weighted F1: {:.4f}%'.format(F1 * 100))

        out_put = np.array(out_put)
        y_true = np.array(y_true)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion Matrix:')
        print(cm)
