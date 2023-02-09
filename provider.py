import os
import sys
import numpy as np
import cupy as cp
import h5py
import yaml
import random
import trimesh

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = cp.zeros(batch_data.shape, dtype=cp.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = cp.random.uniform() * 2 * cp.pi
        cosval = cp.cos(rotation_angle)
        sinval = cp.sin(rotation_angle)
        rotation_matrix = cp.array([[cosval.get(), 0, sinval.get()],
                                    [0, 1, 0],
                                    [-sinval.get(), 0, cosval.get()]])
        shape_pc = batch_data[k, ...]
        shape_pc = cp.array(shape_pc)
        rotated_data[k, ...] = cp.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = cp.zeros(batch_data.shape, dtype=cp.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = cp.cos(rotation_angle)
        sinval = cp.sin(rotation_angle)
        rotation_matrix = cp.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = cp.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = cp.clip(sigma * cp.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    h5_filename = os.path.join(BASE_DIR, h5_filename)
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def load_h5_data_label_seg(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = f['pid'][:]
    return (data, label, seg)


def loadDataFile_with_seg(filename):
    return load_h5_data_label_seg(filename)

def load_uoy_dataset(path, num_point):

    CUTOFF_RACIAL_X = 0.3
    CUTOFF_RATIAL_Z = 0.2

    if 1 and os.path.exists(path + '/configs/pre_processed_data.npy') and os.path.exists(path + '/configs/pre_processed_labels.npy'):
        print('Loading pre-processed data...')
        data = np.load(path + '/configs/pre_processed_data.npy')
        labels = np.load(path + '/configs/pre_processed_labels.npy')
        return data, labels

    with open(path + '/configs/config.yaml', 'r') as f:
        configs = yaml.load(f, Loader = yaml.FullLoader)
    filelist = configs['files']

    with open(path + '/configs/labels.yaml', 'r') as f:
        labels = yaml.load(f, Loader = yaml.FullLoader)
    labels = labels['labels']

    data, size = [], []
    # load model files and convert to point clouds
    total = len(filelist)
    current = 0
    last = progress_bar(current, total, 0, 'Preparing data...')
    for file in filelist:
        mesh = trimesh.load(path + '/models/' + file + '.obj')
        vertices = mesh.vertices

        # remove points outside the cutoff (x: 30% from the most negative, z: 20% from the most positive)
        xs, zs = np.array(vertices[:, 2]), np.array(vertices[:, 1])
        cutoff_x = xs.min() + ((xs.max() - xs.min()) * CUTOFF_RACIAL_X)
        cutoff_z = zs.max() - ((zs.max() - zs.min()) * CUTOFF_RATIAL_Z)
        vertices = vertices[(vertices[:, 2] > cutoff_x) & (vertices[:, 1] < cutoff_z)]

        np.random.shuffle(vertices)
        data.append(np.array(vertices, dtype = np.float32))
        size.append(len(vertices))
        current += 1
        last = progress_bar(current, total, last, 'Preparing data...')

    print(' Done.')

    if num_point > min(size):
        raise ValueError('num_point should be smaller than the minimum point cloud size')

    # slice point clouds to the same size
    for i in range(len(data)):
        # data[i] = data[i][:num_point, :]
        data[i] = data[i][np.random.choice(data[i].shape[0], num_point, replace=False), :]
    
    data, labels = np.array(data), np.array(labels)

    np.save(path + '/configs/pre_processed_data.npy', data)
    np.save(path + '/configs/pre_processed_labels.npy', labels)

    return data, labels

def shuffle_UOY(data, labels):
    
    # slice 20% of the data as test set
    test_num = int(len(data) * 0.2) # approximately 100 data in the training set
    slices = random.sample(range(len(data)), test_num)
    slices.sort()
    test_data = [data[i] for i in slices]
    test_label = [labels[i] for i in slices]
    remainings = [i for i in range(len(data)) if i not in slices]
    np.random.shuffle(remainings)
    train_data = [data[i] for i in remainings]
    train_label = [labels[i] for i in remainings]

    return np.array(train_data), np.array(train_label), np.array(test_data), np.array(test_label)

def progress_bar(current, total, last, msg):
    percentage = round(current / total, 2) * 100
    if current == 0:
        print (msg)
        print ('Progress: |                                                  |  0%', end = '', flush = True) 
    if percentage <= last:
        return last
    percentage_str = str(percentage).split('.')[0]
    i = 0
    while i <= 10:
        print ('\b\b\b\b\b', end = '')
        i += 1
    for i in range(0, int(percentage_str) // 2):
        print ('=', end = '')
    for i in range(int(percentage_str) // 2, 50):
        print (' ', end = '')
    print ('|', end = '')
    if percentage < 10:
        print ('  ' + percentage_str + '%', end = '', flush = True)
    else:
        print (' ' + percentage_str + '%', end = '', flush = True)
    return percentage