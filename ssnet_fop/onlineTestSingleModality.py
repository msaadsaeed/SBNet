"""
"Fusion and Orthogonal Projection for Improved Face-Voice Association"
Muhammad Saad Saeed and Muhammad Haris Khan and Shah Nawaz and Muhammad Haroon Yousaf and Alessio Del Bue
ICASSP 2022
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.autograd import Variable

import pandas as pd
from sklearn import metrics
# from scipy.optimize import brentq
from sklearn.model_selection import KFold
from scipy import interpolate

def read_data(FLAGS):
    face = 'facenet'
    if FLAGS.split_type == 'voice_only':
        print('Reading Voice Test Anchors')
        test_file_anc = 'D:/research/ssnet/voxTrainTestData/voice/twoBranchVoiceOnlyAnchor.csv'
        test_anc = pd.read_csv(test_file_anc, header=None)
        print('Reading Voice Test PosNeg')
        test_file_neg = 'D:/research/ssnet/voxTrainTestData/voice/twoBranchVoiceOnlyPosNeg.csv'
        test_neg = pd.read_csv(test_file_neg, header=None)
        
        test_anc = np.asarray(test_anc)
        test_neg = np.asarray(test_neg)
    
    elif FLAGS.split_type == 'face_only':
        print('Reading Face Test Anchors')
        test_file_anc = 'D:/research/ssnet/voxTrainTestData/faces/%s_AncFaceTest_random_unseenunheard.csv'%(face)
        test_anc = pd.read_csv(test_file_anc, header=None)
        print('Reading Face Test PosNeg')
        test_file_neg = 'D:/research/ssnet/voxTrainTestData/faces/facenet_face_veriflist_test_random_unseenunheard.csv'
        test_neg = pd.read_csv(test_file_neg, header=None)
        
        test_anc = np.asarray(test_anc)
        test_neg = np.asarray(test_neg)
    
    test_list = []
    for dat in range(len(test_anc)):
        test_list.append(test_anc[dat])
        test_list.append(test_neg[dat])
        
    test_list = np.asarray(test_list)
    test_list = torch.from_numpy(test_list).float()
    
    return test_list


# In[1]

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

def test(args, model, test_feat):

    model.eval()
    model.cuda()
 
    if args.cuda:
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
    
        print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    
        auc = metrics.auc(fpr, tpr)
        print('Area Under Curve (AUC): %1.3f' % auc)
        fnr = 1-tpr
        abs_diffs = np.abs(fpr-fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        # eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    #    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
        print('Equal Error Rate (EER): %1.3f\n\n' % eer)
    
    return eer, auc
