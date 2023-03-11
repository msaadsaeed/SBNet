
from __future__ import division
from __future__ import print_function
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import argparse
import sys

import numpy as np
import tensorflow as tf

import online_evaluation as eval_
import csv
from sklearn import preprocessing
from scipy import random
import pandas as pd
import progressbar
from glob import glob

FLAGS = None


# In[0]:
def readData(FLAGS):
    
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
        train_file_face = '../data/face/facenetFaceTrain.csv'
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
    
    # train_data = np.asarray(train_data).astype(np.float)
    train_label = np.asarray(train_label)
    
    return train_data, train_label



# In[1]:

# print('Training\n\n')
# import retrieval_model_sep_loss
from retrieval_model import setup_train_model

tf.reset_default_graph()
def get_batch(batch_index, batch_size, labels, f_lst):
    start_ind = batch_index * batch_size
    end_ind = (batch_index + 1) * batch_size
    return np.asarray(f_lst[start_ind:end_ind]), np.asarray(labels[start_ind:end_ind])

def read_file_org(file_name):
    feat_lst = []
    label_lst = []
    with open(file_name) as fr:
        reader = csv.reader(fr, delimiter=',')
        for row in reader:
            class_label = int(float(row[-1])) - 1
            #print(class_label)
            row = row[:-1]
            s_feat = [float(i) for i in row]
            feat_lst.append(s_feat)
            label_lst.append(class_label)

    return  feat_lst, label_lst

def read_file_org_test(file_name):
    feat_lst = []
    label_lst = []
    count = 0
    with open(file_name) as fr:
        reader = csv.reader(fr, delimiter=',')
        for row in reader:
            s_feat = [float(i) for i in row]
            feat_lst.append(s_feat)
            label_lst.append(count)
    return feat_lst, label_lst

def main(train_data, train_label, test_data, args):
    
    save_dir = '%s_fc2_%s/'%(FLAGS.loss, FLAGS.split_type)
    best = 'best_%s'%(save_dir)
    if not os.path.exists(best):
        os.mkdir(best)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    feat_dim = 512
    
    print('Training')

    # Setup placeholders for input variables.
    feat_plh = tf.placeholder(tf.float32, shape=(None, feat_dim))
    label_plh = tf.placeholder(tf.int64, shape=(None), name='labels')
    train_phase_plh = tf.placeholder(tf.bool)

    # Setup training operation.
    embeds, total_loss = setup_train_model(feat_plh, train_phase_plh, label_plh, FLAGS)
    # Setup optimizer.
    global_step = tf.Variable(0, trainable=False)
    init_learning_rate = 0.00001
    optim1 = tf.train.AdamOptimizer(init_learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    with tf.control_dependencies(update_ops):
        train_step = optim1.minimize(total_loss, global_step=global_step)

    # Setup model saver.
    saver = tf.train.Saver(save_relative_paths=False, max_to_keep=100)

    num_train_samples = len(train_data)
    num_of_batches = (num_train_samples // FLAGS.batch_size)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    soft_loss = []
    # cent_loss = []
    # count = 0
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    eer_list = []
    sess = tf.Session(config=config)
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(100):
        print('Epoch: %03d'%(i))
        bar = progressbar.ProgressBar(max_value=num_of_batches)
        for idx in range(num_of_batches):
            # im_feats, batch_labels = get_batch(idx, FLAGS.batch_size, train_label, img_train)
            # sent_feats, _ = get_batch(idx, FLAGS.batch_size, train_label, voice_train)
            train_batch, batch_labels = get_batch(idx, FLAGS.batch_size, train_label, train_data)
            feed_dict = {
                    feat_plh : train_batch,    
                    label_plh : batch_labels,
                    train_phase_plh : True,
            }
            _, soft = sess.run([train_step, total_loss], feed_dict = feed_dict)
            
            soft_loss.append(soft)
            bar.update(idx)
            
        
        feed_dict = {
                    feat_plh : test_data,
                    train_phase_plh : False,
            }

        test_embeds = sess.run([embeds], feed_dict = feed_dict)
        test_embeds = np.asarray(test_embeds).squeeze()
        print(FLAGS.split_type)
        saver.save(sess, save_dir, global_step = global_step)
        eer, auc = eval_.eval_train(test_embeds)
        eer_list.append(eer)
        if eer <= min(eer_list):
            prev = glob(best+'*')
            for tmp_ckpt in prev:
                os.remove(tmp_ckpt)
            saver.save(sess, best, global_step = global_step)
            min_eer = eer
            max_auc = auc
            best_epoch = i
            
    print('Best Epoch: %d'%(best_epoch))
    print('EER: %0.3f'%(min_eer))
    print('AUC: %0.3f'%(max_auc))


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)
    parser = argparse.ArgumentParser()
    # Training parameters.
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--max_num_epoch', type=int, default=50, help='Max number of epochs to train.')
    parser.add_argument('--split_type', type=str, default='hefhev', help='split_type')
    parser.add_argument('--loss', type=str, default='git', help='Loss: [git, cent]')
    FLAGS, unparsed = parser.parse_known_args()
    train_data, train_label = readData(FLAGS)
    test_embeds = eval_.read_data(FLAGS)
    main(train_data, train_label, test_embeds, sys.argv)
    