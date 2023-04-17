import argparse 
import math
import h5py
import numpy as np
import cupy as cp
import socket
import importlib
import matplotlib.pyplot as plt
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import provider
import random
import utils.data_utils
import time

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


from utils.model import RandPointCNN
from utils.util_funcs import knn_indices_func_gpu, knn_indices_func_cpu
from utils.util_layers import Dense


random.seed(0)
dtype = torch.cuda.FloatTensor

# config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2000, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.8, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

OPTIMIZER = FLAGS.optimizer

NUM_POINT = FLAGS.num_point
LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
       
MAX_NUM_POINT = 2048

DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


LEARNING_RATE_MIN = 0.00001
        
# NUM_CLASS = 40
NUM_CLASS = 10
BATCH_SIZE = FLAGS.batch_size #32
NUM_EPOCHS = FLAGS.max_epoch
jitter = 0.01
jitter_val = 0.01

rotation_range = [0, math.pi / 18, 0, 'g']
rotation_rage_val = [0, 0, 0, 'u']
order = 'rxyz'

scaling_range = [0.05, 0.05, 0.05, 'g']
scaling_range_val = [0, 0, 0, 'u']

class modelnet40_dataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]


# C_in, C_out, D, N_neighbors, dilution, N_rep, r_indices_func, C_lifted = None, mlp_width = 2
# (a, b, c, d, e) == (C_in, C_out, N_neighbors, dilution, N_rep)
# Abbreviated PointCNN constructor.
AbbPointCNN = lambda a, b, c, d, e: RandPointCNN(a, b, 3, c, d, e, knn_indices_func_gpu)


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()

        self.pcnn1 = AbbPointCNN(3, 32, 8, 1, -1)
        self.pcnn2 = nn.Sequential(
            AbbPointCNN(32, 64, 8, 2, -1),
            AbbPointCNN(64, 96, 8, 4, -1),
            AbbPointCNN(96, 128, 12, 4, 120),
            AbbPointCNN(128, 160, 12, 6, 120)
        )

        self.fcn = nn.Sequential(
            Dense(160, 128),
            Dense(128, 64, drop_rate=0.5),
            Dense(64, NUM_CLASS, with_bn=False, activation=None)
        )

    def forward(self, x):
        x = self.pcnn1(x)
        if False:
            print("Making graph...")
            k = make_dot(x[1])

            print("Viewing...")
            k.view()
            print("DONE")

            assert False
        x = self.pcnn2(x)[1]  # grab features

        logits = self.fcn(x)
        logits_mean = torch.mean(logits, dim=1)
        return logits_mean


print("------Building model-------")
model = Classifier().cuda()
print("------Successfully Built model-------")

# determine which optimizer to use. Default sgd
if OPTIMIZER == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum = 0.9)

loss_fn = nn.CrossEntropyLoss()

global_step = 1

#model_save_dir = os.path.join(CURRENT_DIR, "models", "mnist2")
#os.makedirs(model_save_dir, exist_ok = True)

# TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
# TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

data, labels = provider.load_uoy_dataset(os.path.join(BASE_DIR, 'data/UOYFaces'), NUM_POINT)

prefix = time.strftime('%Y_%m_%d', time.localtime())
suffix = '_' + str(LEARNING_RATE) + '_' + str(BATCH_SIZE) + '_' + str(NUM_EPOCHS) + '_' + OPTIMIZER
DATA_SAVE_DIR = '/home/uqhyan11/xconv/PointCNN.Pytorch-master/results/'

losses = []
test_accs = []
train_accs = []
lrs = [LEARNING_RATE]
epoch_count = []
epoch_losses = []

'''
if False:
    latest_model = sorted(os.listdir(model_save_dir))[-1]
    model.load_state_dict(torch.load(os.path.join(model_save_dir, latest_model)))    
'''

# data = []
# labels = []
# for i in range(len(TRAIN_FILES)):
#     current_data, current_labels = provider.loadDataFile(TRAIN_FILES[i])
#     data.append(current_data)
#     labels.append(current_labels)

last = provider.progress_bar(0, NUM_EPOCHS, 0, 'Training...')

# current_data has the structure:
# 3-dimensional nd.array(float32)
# 1st dimension: the models
# 2nd dimension: points in each model
# 3rd dimension: XYZ coor of each point

# current_label has the structure:
# 2-dimensional nd.array(uint8)
# 1st dimension: the models
# 2nd dimension: the class 
current_data, current_label, test_data, test_label = provider.shuffle_UOY(data, labels)

# preparing test set in advance
T_sampled = torch.tensor(test_data).float()
T_sampled = Variable(T_sampled, requires_grad=False).cuda()
T_label = torch.from_numpy(test_label).long()
T_label = Variable(T_label, requires_grad=False).cuda()
TEST_DATASET_SIZE = len(test_label)

for epoch in range(1, NUM_EPOCHS+1):
    train_acc = []
    last = provider.progress_bar(epoch, NUM_EPOCHS, last, 'Training...')
    epoch_count.append(epoch)

    #log_string('----' + str(fn) + '-----')
    # print('Start loading data at: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    # start_t = time.time()

    # current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
    # current_label = np.squeeze(current_label)

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    # print('End loading data at: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    # print('Total time: ', time.time() - start_t)

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    if epoch > 1:
        LEARNING_RATE *= DECAY_RATE ** (global_step // DECAY_STEP)
        if LEARNING_RATE > LEARNING_RATE_MIN:
            # print("NEW LEARNING RATE:", LEARNING_RATE, 'Calculated by: ', LEARNING_RATE, ' * ', DECAY_RATE, ' ** ', global_step, ' // ', DECAY_STEP)
            if OPTIMIZER == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum = 0.9)
            lrs.append(LEARNING_RATE)
            with open(DATA_SAVE_DIR + prefix + suffix + "_lr.txt", "a+") as f:
                f.write('Epoch: ' + str(epoch) + ' Lr: ' + str(LEARNING_RATE) + ' DECAY_RATE: ' + str(DECAY_RATE) + ' global_step: ' + str(global_step) + ' DECAY_STEP: ' + str(DECAY_STEP) + "\n")
                f.close()

    for batch_idx in range(num_batches):
        losses = []
        # print ('batch starts at ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        batch_t = time.time()
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        # Lable
        label = current_label[start_idx:end_idx]
        TRAIN_SIZE = len(label)
        label = torch.from_numpy(label).long()
        label = Variable(label, requires_grad=False).cuda()
        # Augment batched point clouds by rotation and jittering
        # print('Start augmenting data at: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        # rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
        # jittered_data = provider.jitter_point_cloud(rotated_data) # P_Sampled
        # P_sampled = jittered_data
        P_sampled = current_data[start_idx:end_idx, :, :]
        F_sampled = cp.zeros((BATCH_SIZE, NUM_POINT, 0))
        optimizer.zero_grad()

        # t0 = time.time()
        P_sampled = torch.tensor(P_sampled).float()
        P_sampled = Variable(P_sampled, requires_grad=False).cuda()
        # F_sampled = torch.from_numpy(F_sampled)

        out = model((P_sampled, P_sampled))
        loss = loss_fn(out, label)
        loss.backward()

        optimizer.step()
        # print("epoch: "+str(epoch) + "   loss: "+str(loss.item()))
        losses.append(loss.item())
        if global_step % 25 == 0:
            loss_v = loss.item()
            # print("Loss:", loss_v)
        else:
            loss_v = 0
        global_step += 1

        correct = out.data.max(1)[1].eq(label.data).cpu().sum()
        train_acc.append(correct.item() / float(TRAIN_SIZE))
        torch.cuda.empty_cache()

        # print('Batch ends at: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), 'total time: ', time.time() - batch_t)
    # calcilate train accuracy in each epoch
    train_accs.append(np.mean(train_acc))

    # calculate test accuracy in each epoch
    test_out = model((T_sampled, T_sampled))
    pred_label = test_out.data.max(1)[1]
    # pred_label = torch.max(test_out.data, 1)[1]
    correct = pred_label.eq(T_label.data).cpu().sum()
    acc = correct.item() / float(TEST_DATASET_SIZE)
    test_accs.append(acc)

    batch_loss = np.mean(losses)
    with open(DATA_SAVE_DIR + prefix + suffix + "_losses.txt", "a+") as f:
        f.write('Epoch: ' + str(epoch) + ' loss: ' + str(batch_loss) + ' Acc: ' + str(acc) + "\n")
        f.close()

    epoch_losses.append(batch_loss)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(epoch_count, epoch_losses, 'r--', label = 'Training Loss')
    ax2.plot(epoch_count, test_accs, 'b--', label = 'Test Accuracy')
    ax2.plot(epoch_count, train_accs, 'g--', label = 'Train Accuracy')
    ax.title.set_text('Training Results')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')
    ax.legend()
    ax2.legend()
    fig.savefig(DATA_SAVE_DIR + prefix + suffix + "_results.png")
    plt.close(fig)
print(' Done.')


# output losses, acc, lr
# prefix = time.strftime('%Y_%m_%d', time.localtime())
# with open('./results/' + prefix + "losses.txt", "w") as f:
#     for i in losses:
#         f.write(str(i) + "\n")
#     f.close()
# with open('./results/' + prefix + "lrs.txt", "w") as f:
#     for i in lrs:
#         f.write(str(i) + '\n')
#     f.close()