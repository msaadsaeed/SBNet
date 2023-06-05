
from __future__ import division
from __future__ import print_function

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import pandas as pd
from scipy import random
from sklearn import preprocessing
# import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm


# In[0]

def read_data(FLAGS):
    
    print('Split Type: %s'%(FLAGS.split_type))
    
    if FLAGS.split_type == 'voice_only':
        print('Reading Voice Train')
        train_file_voice = '../data/voice/voiceTrain.csv'
        train_data = pd.read_csv(train_file_voice, header=None)
        train_label = train_data[512]
        le = preprocessing.LabelEncoder()
        le.fit(train_label)
        train_label = le.transform(train_label)
        train_data = np.asarray(train_data)
        train_data = train_data[:, :-1]
        
        return train_data, train_label
        
    elif FLAGS.split_type == 'face_only':
        print('Reading Face Train')
        train_file_face = '../data/face/facenetfaceTrain.csv'
        train_data = pd.read_csv(train_file_face, header=None)
        train_label = train_data[512]
        le = preprocessing.LabelEncoder()
        le.fit(train_label)
        train_label = le.transform(train_label)
        train_data = np.asarray(train_data)
        train_data = train_data[:, :-1]
        
        return train_data, train_label
    
    train_data = []
    train_label = []
    
    train_file_face = '../data/face/facenetfaceTrain.csv'
    train_file_voice = '../data/voice/voiceTrain.csv'
    
    print('Reading Train Faces')
    img_train = pd.read_csv(train_file_face, header=None)
    train_tmp = img_train[512]
    img_train = np.asarray(img_train)
    img_train = img_train[:, :-1]
    
    train_tmp = np.asarray(train_tmp)
    train_tmp = train_tmp.reshape((train_tmp.shape[0], 1))
    print('Reading Train Voices')
    voice_train = pd.read_csv(train_file_voice, header=None)
    voice_train = np.asarray(voice_train)
    voice_train = voice_train[:, :-1]
    
    combined = list(zip(img_train, voice_train, train_tmp))
    random.shuffle(combined)
    img_train, voice_train, train_tmp = zip(*combined)
    
    if FLAGS.split_type == 'random':
        train_data = np.vstack((img_train, voice_train))
        train_label = np.vstack((train_tmp, train_tmp))
        combined = list(zip(train_data, train_label))
        random.shuffle(combined)
        train_data, train_label = zip(*combined)
        train_data = np.asarray(train_data).astype(np.float)
        train_label = np.asarray(train_label)
    
    elif FLAGS.split_type == 'vfvf':
        for i in range(len(voice_train)):
            train_data.append(voice_train[i])
            train_data.append(img_train[i])
            train_label.append(train_tmp[i])
            train_label.append(train_tmp[i])
            
    elif FLAGS.split_type == 'fvfv':
        for i in range(len(voice_train)):
            train_data.append(img_train[i])
            train_data.append(voice_train[i])
            train_label.append(train_tmp[i])
            train_label.append(train_tmp[i])        
    
    elif FLAGS.split_type == 'hefhev':
        train_data = np.vstack((img_train, voice_train))
        train_label = np.vstack((train_tmp, train_tmp))
        
    elif FLAGS.split_type == 'hevhef':
        train_data = np.vstack((voice_train, img_train))
        train_label = np.vstack((train_tmp, train_tmp))
    
    else:
        print('Invalid Split Type')
    
    le = preprocessing.LabelEncoder()
    le.fit(train_label)
    train_label = le.transform(train_label)
    
    print("Train file length", len(img_train))
    print('Shuffling\n')
    
    train_data = np.asarray(train_data).astype(np.float)
    train_label = np.asarray(train_label)
    
    return train_data, train_label


# In[1]
from retrieval_model import FOP
 
def get_batch(batch_index, batch_size, labels, f_lst):
    start_ind = batch_index * batch_size
    end_ind = (batch_index + 1) * batch_size
    return np.asarray(f_lst[start_ind:end_ind]), np.asarray(labels[start_ind:end_ind])

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

def main(train_data, train_label):
    
    n_class = 901
    model = FOP(FLAGS, train_data.shape[1], n_class)
    model.apply(init_weights)
    
    ce_loss = nn.CrossEntropyLoss().cuda()
    opl_loss = OrthogonalProjectionLoss().cuda()
    
    if FLAGS.cuda:
        model.cuda()
        ce_loss.cuda()    
        opl_loss.cuda()
        cudnn.benchmark = True
    
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=0.01)

    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))
    
    
    for alpha in FLAGS.alpha_list:
        eer_list = []
        epoch=1
        num_of_batches = (len(train_label) // FLAGS.batch_size)
        loss_plot = []
        auc_list = []
        loss_per_epoch = 0
        s_fac_per_epoch = 0
        d_fac_per_epoch = 0
        txt_dir = 'output'
        save_dir = 'fc2_%s_%s_alpha_%0.2f'%(FLAGS.split_type, FLAGS.save_dir, alpha)
        txt = '%s/ce_opl_%03d_%0.2f.txt'%(txt_dir, FLAGS.max_num_epoch, alpha)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if not os.path.exists(txt_dir):
            os.makedirs(txt_dir)
        
        with open(txt,'w+') as f:
            f.write('EPOCH\tLOSS\tEER\tAUC\tS_FAC\tD_FAC\n')
        
        save_best = 'best_%s'%(save_dir)
        
        if not os.path.exists(save_best):
            os.mkdir(save_best)
        with open(txt,'a+') as f:
            while (epoch < FLAGS.max_num_epoch):
                print('%s\tEpoch %03d'%(FLAGS.split_type, epoch))
                for idx in tqdm(range(num_of_batches)):
                    train_batch, batch_labels = get_batch(idx, FLAGS.batch_size, train_label, train_data)
                    # voice_feats, _ = get_batch(idx, FLAGS.batch_size, train_label, voice_train)
                    loss_tmp, loss_opl, loss_soft, s_fac, d_fac = train(train_batch, 
                                                                 batch_labels, 
                                                                 model, optimizer, ce_loss, opl_loss, alpha)
                    loss_per_epoch+=loss_tmp
                    s_fac_per_epoch+=s_fac
                    d_fac_per_epoch+=d_fac
                
                loss_per_epoch/=num_of_batches
                s_fac_per_epoch/=num_of_batches
                d_fac_per_epoch/=num_of_batches
                
                loss_plot.append(loss_per_epoch)
                if FLAGS.split_type == 'voice_only' or FLAGS.split_type == 'face_only':
                    eer, auc = onlineTestSingleModality.test(FLAGS, model, test_feat)
                else:
                    eer, auc = online_evaluation.test(FLAGS, model, test_feat)
                eer_list.append(eer)
                auc_list.append(auc)
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict()}, save_dir, 'checkpoint_%04d_%0.3f.pth.tar'%(epoch, eer*100))

                print('==> Epoch: %d/%d Loss: %0.2f Alpha:%0.2f, Min_EER: %0.2f'%(epoch, FLAGS.max_num_epoch, loss_per_epoch, alpha, min(eer_list)))
                
                if eer <= min(eer_list):
                    min_eer = eer
                    max_auc = auc
                    save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict()}, save_best, 'checkpoint.pth.tar')
            
                f.write('%04d\t%0.4f\t%0.2f\t%0.2f\t%0.2f\t%0.2f\n'%(epoch, loss_per_epoch, eer, auc, s_fac_per_epoch, d_fac_per_epoch))
                loss_per_epoch = 0
                s_fac_per_epoch = 0
                d_fac_per_epoch = 0
                epoch += 1
        
                
        return loss_plot, min_eer, max_auc
    
class OrthogonalProjectionLoss(nn.Module):
    def __init__(self):
        super(OrthogonalProjectionLoss, self).__init__()
        self.device = (torch.device('cuda') if FLAGS.cuda else torch.device('cpu'))

    def forward(self, features, labels=None):
        
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]

        mask = torch.eq(labels, labels.t()).bool().to(self.device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(self.device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)

        loss = (1.0 - pos_pairs_mean) + (0.7 * neg_pairs_mean)

        return loss, pos_pairs_mean, neg_pairs_mean


def train(train_batch, labels, model, optimizer, ce_loss, opl_loss, alpha):
    
    average_loss = RunningAverage()
    soft_losses = RunningAverage()
    opl_losses = RunningAverage()

    model.train()
    # face_feats = torch.from_numpy(face_feats).float()
    train_batch = torch.from_numpy(train_batch).float()
    labels = torch.from_numpy(labels)
    
    if FLAGS.cuda:
        train_batch, labels = train_batch.cuda(), labels.cuda()

    train_batch, labels = Variable(train_batch), Variable(labels)
    comb = model.train_forward(train_batch)
    
    loss_opl, s_fac, d_fac = opl_loss(comb[0], labels)
    
    loss_soft = ce_loss(comb[1], labels)
    
    loss = loss_soft + alpha * loss_opl

    optimizer.zero_grad()
    
    loss.backward()
    average_loss.update(loss.item())
    opl_losses.update(loss_opl.item())
    soft_losses.update(loss_soft.item())
    
    optimizer.step()

    return average_loss.avg(), opl_losses.avg(), soft_losses.avg(), s_fac, d_fac

class RunningAverage(object):
    def __init__(self):
        self.value_sum = 0.
        self.num_items = 0. 

    def update(self, val):
        self.value_sum += val 
        self.num_items += 1

    def avg(self):
        average = 0.
        if self.num_items > 0:
            average = self.value_sum / self.num_items

        return average
 
def save_checkpoint(state, directory, filename):
    filename = os.path.join(directory, filename)
    torch.save(state, filename)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random Seed')
    parser.add_argument('--cuda', action='store_true', default=True, help='CUDA Training')
    parser.add_argument('--save_dir', type=str, default='model', help='Directory for saving checkpoints.')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-4)') 
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--max_num_epoch', type=int, default=500, help='Max number of epochs to train, number')
    parser.add_argument('--alpha_list', type=list, default=[1], help='Alpha Values List')
    parser.add_argument('--dim_embed', type=int, default=128,
                        help='Embedding Size')
    parser.add_argument('--split_type', type=str, default='hefhev', help='split_type')

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    torch.manual_seed(FLAGS.seed)
    if FLAGS.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)
    if FLAGS.split_type == 'voice_only' or FLAGS.split_type == 'face_only':
        import onlineTestSingleModality
        test_feat = onlineTestSingleModality.read_data(FLAGS)
    else:
        import online_evaluation
        test_feat = online_evaluation.read_data()
    train_data, train_label = read_data(FLAGS)
    
    main(train_data, train_label)
