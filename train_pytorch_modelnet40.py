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



# Load Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls',
                    help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=6000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.8, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

OPTIMIZER = FLAGS.optimizer

NUM_POINT = FLAGS.num_point
LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
       
MAX_NUM_POINT = FLAGS.num_point

DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99


LEARNING_RATE_MIN = 0.00001
        
NUM_CLASS = 40
BATCH_SIZE = FLAGS.batch_size #32
NUM_EPOCHS = FLAGS.max_epoch
jitter = 0.01
jitter_val = 0.01

rotation_range = [0, math.pi / 18, 0, 'g']
rotation_rage_val = [0, 0, 0, 'u']
order = 'rxyz'

scaling_range = [0.05, 0.05, 0.05, 'g']
scaling_range_val = [0, 0, 0, 'u']

class modelnet40(Dataset):
    def __init__(self, data, labels, transform = None):
        self.data = data
        self.labels = torch.tensor(labels).long()
        self.transform = transform
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        # if self.transform is not None:
        #     temp = self.transform[0](self.data[i])
        #     temp = self.transform[1](temp)
        #     return temp, self.labels[i]
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


print("Building model... ", end = '')
model = Classifier().cuda()
print('Done')

# determine which optimizer to use. Default sgd
if OPTIMIZER == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum = 0.9)

loss_fn = nn.CrossEntropyLoss()

global_step = 1

#model_save_dir = os.path.join(CURRENT_DIR, "models", "mnist2")
#os.makedirs(model_save_dir, exist_ok = True)

TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
prefix = time.strftime('%Y_%m_%d', time.localtime())
suffix = '_' + str(LEARNING_RATE) + '_' + str(BATCH_SIZE) + '_' + str(NUM_EPOCHS) + '_' + OPTIMIZER
DATA_SAVE_DIR = './results/modelnet40/'

train_accs = []
test_accs = []
lrs = [LEARNING_RATE]
epoch_count = []
losses = []

'''
if False:
    latest_model = sorted(os.listdir(model_save_dir))[-1]
    model.load_state_dict(torch.load(os.path.join(model_save_dir, latest_model)))    
'''

print ('Preparing data... ', end = '')
data, label = None, None

# load all training data and convert to a Dataset adn Dataloader
for FILE in TRAIN_FILES:
    if data is None and label is None:
        data, label = provider.loadDataFile(FILE)
        label = np.squeeze(label)
    else:
        data_, label_ = provider.loadDataFile(FILE)
        data = np.concatenate((data, data_), axis = 0)
        label = np.concatenate((label, np.squeeze(label_)), axis = 0)
train_dataset = modelnet40(data, label, [provider.rotate_point_cloud, provider.jitter_point_cloud])
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

# load all test data and convert to a Dataset and Dataloader
data, label = None, None
for FILE in TEST_FILES:
    if data is None and label is None:
        data, label = provider.loadDataFile(FILE)
        label = np.squeeze(label)
    else:
        data_, label_ = provider.loadDataFile(FILE)
        data = np.concatenate((data, data_), axis = 0)
        label = np.concatenate((label, np.squeeze(label_)), axis = 0)
test_dataset = modelnet40(data, label)
test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

print ('Done')
    
last = provider.progress_bar(0, NUM_EPOCHS, 0, 'Training...')

for epoch in range(1, NUM_EPOCHS+1):
    last = provider.progress_bar(epoch, NUM_EPOCHS, last, 'Training...')
    epoch_count.append(epoch)

    if epoch > 1:
        LEARNING_RATE *= DECAY_RATE ** (global_step // DECAY_STEP)
        if LEARNING_RATE > LEARNING_RATE_MIN:
            # print("NEW LEARNING RATE:", LEARNING_RATE)
            if OPTIMIZER == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum = 0.9)
            lrs.append(LEARNING_RATE)
            with open(DATA_SAVE_DIR + prefix + suffix + "_lr.txt", "a+") as f:
                f.write('Epoch: ' + str(epoch) + ' Lr: ' + str(LEARNING_RATE) + ' DECAY_RATE: ' + str(DECAY_RATE) + ' global_step: ' + str(global_step) + ' DECAY_STEP: ' + str(DECAY_STEP) + "\n")
                f.close()

    batch_losses = []
    batch_accs = []
    for batch_idx, (data, label) in enumerate(train_dataloader):
        processed_data = provider.rotate_point_cloud(data)
        processed_data = provider.jitter_point_cloud(processed_data)
        processed_data = torch.tensor(processed_data).float()
        processed_data = Variable(processed_data, requires_grad = False)
        optimizer.zero_grad()
        out = model((processed_data.cuda(), processed_data.cuda()))
        loss = loss_fn(out, label.cuda())
        loss.backward()
        batch_losses.append(loss.item())
        correct = out.data.max(1)[1].eq(label.cuda()).cpu().sum()
        batch_accs.append(correct.item() / float(len(data)))
        torch.cuda.empty_cache()
    
    losses.append(np.mean(batch_losses))
    train_accs.append(np.mean(batch_accs))

    batch_acc = []
    for batch_idx, (data, label) in enumerate(test_dataloader):
        out = model((data.cuda(), data.cuda()))
        correct = out.data.max(1)[1].eq(label.cuda()).cpu().sum()
        batch_accs.append(correct.item() / float(len(data)))
        torch.cuda.empty_cache()
    
    test_accs.append(np.mean(batch_accs))

    with open(DATA_SAVE_DIR + prefix + suffix + "_losses.txt", "a+") as f:
        f.write('Epoch: ' + str(epoch) + ' loss: ' + str(np.mean(batch_losses)) + ' Train Acc: ' + str(train_accs[-1]) + ' Test Acc: ' + str(test_accs[-1]) + "\n")
        f.close()

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(epoch_count, losses, 'r--', label = 'Training Loss')
    ax2.plot(epoch_count, train_accs, 'b--', label = 'Training Accuracy')
    ax2.plot(epoch_count, test_accs, 'g--', label = 'Test Accuracy')
    ax.title.set_text('Training Results')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')
    ax.legend()
    ax2.legend()
    fig.savefig(DATA_SAVE_DIR + prefix + suffix + "_Results.png")
    plt.close(fig)


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