
from __future__ import division
from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import KFold
from scipy import interpolate


def read_data(FLAGS):
    
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
    
    
    test_feat = torch.from_numpy(test_list).float()
    
    return test_feat



# In[1]

from retrieval_model import FOP

def load_checkpoint(model, resume_filename):
    start_epoch = 1
    best_acc = 0.0
    if resume_filename:
        if os.path.isfile(resume_filename):
            print("=> loading checkpoint '{}'".format(resume_filename))
            checkpoint = torch.load(resume_filename)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume_filename))

    return start_epoch, best_acc

def same_func(f):
    issame_lst = []
    for idx in range(len(f)):
        if idx % 2 == 0:
            issame = True
        else:
            issame = False
        issame_lst.append(issame)
    return issame_lst

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

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

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

def evaluate(embeddings, actual_issame, nrof_folds=10):
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
                                       np.asarray(actual_issame), nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
                                      np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)
    
    print('\nEvaluating')
    return tpr, fpr, accuracy, val, val_std, far

def test(test_feat):
    
    n_class = 901
    model = FOP(FLAGS, test_feat.shape[1], n_class)
    restore_path = 'final_models/fop_facenet_fc2_%s/checkpoint.pth.tar'%(FLAGS.split_type)
    checkpoint = torch.load(restore_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format('checkpoint.pth.tar', checkpoint['epoch']))
    model.eval()
    model.cuda()
    
    if FLAGS.cuda:
        test_feat = test_feat.cuda()

    test_feat = Variable(test_feat)
    
    with torch.no_grad():
        feat_list, _ = model(test_feat)
        
        feat_list = feat_list.data
        
        feat_list = feat_list.cpu().detach().numpy()
    
        print('Total Number of Samples: ', len(feat_list))
        
        issame_lst = same_func(feat_list)
        feat_list = np.asarray(feat_list)
        
        tpr, fpr, accuracy, val, val_std, far = evaluate(feat_list, issame_lst, 10)
    
        print('Accuracy: %1.3f+-%0.2f' % (np.mean(accuracy), np.std(accuracy)))
        
        auc = metrics.auc(fpr, tpr)
        print('Area Under Curve (AUC): %0.3f' % auc)
        fnr = 1-tpr
        abs_diffs = np.abs(fpr-fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        # eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        # eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
        print('Equal Error Rate (EER): %0.3f\n\n' % eer) 
    
    return 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False, help='CUDA training')
    parser.add_argument('--dim_embed', type=int, default=128,
                        help='Embedding Size')
    parser.add_argument('--split_type', type=str, default='vfvf', help='split_type: [hefhev, hevhef, random, vfvf, fvfv, face_only, voice_only]')
    parser.add_argument('--sh', type=str, default='unseenunheard', help='unseen_unheard or seen_heard')
    parser.add_argument('--test', type=str, default='random', help='Test Type: [random, G, N, A, GNA]')
    
    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.cuda = torch.cuda.is_available()
    torch.manual_seed(FLAGS.seed)
    if FLAGS.cuda:
        torch.cuda.manual_seed(FLAGS.seed)
    
    test_feat = read_data(FLAGS)
    test(test_feat)