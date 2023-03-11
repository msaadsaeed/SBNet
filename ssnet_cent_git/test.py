from __future__ import division
from __future__ import print_function

import argparse
import csv
import sys
import time

import pandas as pd

import numpy as np
import tensorflow as tf

from retrieval_model import setup_eval_model
from sklearn.model_selection import KFold
from scipy import interpolate
from sklearn import metrics

from scipy.optimize import brentq

FLAGS = None

tf.reset_default_graph()
def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
                                       np.asarray(actual_issame), nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
                                      np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def same_func(f):
    issame_lst = []
    for idx in range(len(f)):
        if idx % 2 == 0:
            issame = True
        else:
            issame = False
        issame_lst.append(issame)
    return issame_lst


def eval_once(saver, placeholders, test_embeds):
    
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    
    restore_path = 'final_models/best_%s_facenet_fc2_%s/'%(FLAGS.loss, FLAGS.split_type)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # Restore latest checkpoint or the given MetaGraph.
        if restore_path.endswith('.meta'):
            ckpt_path = restore_path.replace('.meta', '')
        else:
            ckpt_path = tf.train.latest_checkpoint(restore_path)
        print('Restoring checkpoint', ckpt_path)
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)
        print('Done')
        
        if FLAGS.split_type == 'voice_only':
            print('Reading Voice Test Anchors')
            test_file_anc = '../data/voice/twoBranchVoiceOnlyAnchor.csv'
            test_anc = pd.read_csv(test_file_anc, header=None)
            print('Reading Voice Test PosNeg')
            test_file_neg = '../data/voice/twoBranchVoiceOnlyPosNeg.csv'
            test_neg = pd.read_csv(test_file_neg, header=None)
            
            test_anc = np.asarray(test_anc)
            test_neg = np.asarray(test_neg)
            
            test_list = []
            for dat in range(len(test_anc)):
                test_list.append(test_anc[dat])
                test_list.append(test_neg[dat])
                
            test_list = np.asarray(test_list)
            
        elif FLAGS.split_type == 'face_only':
            print('Reading Face Test Anchors')
            test_file_anc = '../data/face/facenet_AncFaceTest_random_unseenunheard.csv'
            test_anc = pd.read_csv(test_file_anc, header=None)
            print('Reading Face Test PosNeg')
            test_file_neg = '../data/face/facenet_face_veriflist_test_random_unseenunheard.csv'
            test_neg = pd.read_csv(test_file_neg, header=None)
            
            test_anc = np.asarray(test_anc)
            test_neg = np.asarray(test_neg)
            
            test_list = []
            for dat in range(len(test_anc)):
                test_list.append(test_anc[dat])
                test_list.append(test_neg[dat])
                
            test_list = np.asarray(test_list)        
        else:
            test_file_face = '../data/face/facenet_face_veriflist_test_%s_%s.csv'%(FLAGS.test, FLAGS.sh)
            test_file_voice = '../data/voice/voice_veriflist_test_%s_%s.csv'%(FLAGS.test, FLAGS.sh)
        
            print('Reading Test Faces')
            face_test = pd.read_csv(test_file_face, header=None)
            print('Reading Test Voices')
            voice_test = pd.read_csv(test_file_voice, header=None)
            
            face_test = np.asarray(face_test)[:,:512]
            voice_test = np.asarray(voice_test)[:,:512]
            
            test_list = []
            for dat in range(len(voice_test)):
                test_list.append(voice_test[dat])
                test_list.append(face_test[dat])
            
            test_list = np.asarray(test_list)
        

        feed_dict = {
            placeholders['feats']: test_list,
            placeholders['train_phase']: False,
        }
        
        [feat_list] = sess.run([test_embeds], feed_dict=feed_dict)

        print('Total Number of Samples: ', len(feat_list))

        issame_lst = same_func(feat_list)
        feat_list = np.asarray(feat_list)

        tpr, fpr, accuracy, val, val_std, far = evaluate(feat_list, issame_lst, 10)
        
        print('%s'%(FLAGS.split_type))
        print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))

        auc = metrics.auc(fpr, tpr)
        print('Area Under Curve (AUC): %1.3f' % auc)
        fnr = 1-tpr
        abs_diffs = np.abs(fpr-fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        
        # eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
        print('Equal Error Rate (EER): %1.3f' % eer)
        

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


def main(_):
    feat_dim = 512

    # Setup placeholders for input variables.
    feat_plh = tf.placeholder(tf.float32, shape=[None, feat_dim])
    
    train_phase_plh = tf.placeholder(tf.bool)
    placeholders = {
        'feats': feat_plh,
        'train_phase': train_phase_plh,
    }

    # Setup testing operation.
    test_embeds = setup_eval_model(feat_plh, train_phase_plh)

    # Setup checkpoint saver.
    saver = tf.train.Saver(save_relative_paths=True)

    # Periodically evaluate the latest checkpoint in the restore_dir directory,
    # unless a specific chekcpoint MetaGraph path is provided.
    eval_once(saver, placeholders, test_embeds)


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    parser = argparse.ArgumentParser()
    # Dataset and checkpoints.
    parser.add_argument('--split_type', type=str, default='vfvf', help='split_type: [hefhev, hevhef, random, vfvf, fvfv, face_only, voice_only]')
    parser.add_argument('--loss', type=str, default='cent', help='Loss: [cent, git]')
    parser.add_argument('--sh', type=str, default='unseenunheard', help='unseenunheard/seen_heard')
    parser.add_argument('--test', type=str, default='random', help='test split type: [random, G, N, A, GNA]')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
