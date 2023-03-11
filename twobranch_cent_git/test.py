from __future__ import division
from __future__ import print_function

import argparse
import csv
import sys

import numpy as np
import tensorflow as tf

from retrieval_model import setup_eval_model
from sklearn.model_selection import KFold
from scipy import interpolate
from sklearn import metrics


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

def plot_tsne(face, voice):
    import pandas as pd
    from sklearn.manifold import TSNE
    from matplotlib import colors
    import matplotlib.pyplot as plt
    from sklearn import preprocessing
    
    filename = 'D:/learnablePins/faceFilesTest.txt'

    labels = []
    count = 0
    counter = 0
    prev = 10002
    voice_feats = []
    face_feats = []
    with open(filename, 'r+') as f:
        for i, dat in enumerate(f):
            if counter>10:
                break
            if count%2 == 0:
                curr = dat.split('\n')[0].split('/')[0].split('id')[1]
                face_feats.append(face[i])
                voice_feats.append(voice[i])
                labels.append(curr)
                if curr!=prev:
                    prev = curr
                    counter+=1
                count=1
            else:
                count = 0
    labels = labels[:-1]
    # feats = feats[:-1]
    # face_feats = face_feats[:-1]
    # voice_feats = voice_feats[:-1]
    
    face_feats = np.asarray(face_feats)
    voice_feats = np.asarray(voice_feats)

    labels = np.asarray(labels)
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    color = list(colors._colors_full_map.values())
    tsne_embeds = TSNE(n_components=2).fit_transform(voice_feats)
    
    ax = plt.subplot()
    
    groups = pd.DataFrame(tsne_embeds, columns=['x', 'y']).assign(category=labels).groupby('category')
    for name, points in groups:
        ax.scatter(points.x, points.y, marker='o', label=name)
    
    tsne_embeds = TSNE(n_components=2).fit_transform(face_feats)
    
    groups = pd.DataFrame(tsne_embeds, columns=['x', 'y']).assign(category=labels).groupby('category')
    for name, points in groups:
        ax.scatter(points.x, points.y, marker='x')
    
    ax.legend(bbox_to_anchor = (1,1))
    
    # dim1, dim2 = zip(*tsne_embeds)
    # plt.scatter(dim1, dim2, marker = 'o', c=labels, cmap=colors.ListedColormap(color), label = 'voice')
    
    # dim1, dim2 = zip(*tsne_embeds)
    # plt.scatter(dim1, dim2, marker = 'x', c=labels, cmap=colors.ListedColormap(color), label = 'face')
    # plt.legend(loc='upper left')
    # plt.savefig('bbb.png', dpi=1200, bbox_inches = 'tight')
    plt.grid()
    plt.show()


def eval_once(saver, placeholders, feat_img, feat_sent):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        # Restore latest checkpoint or the given MetaGraph.
        if FLAGS.restore_path.endswith('.meta'):
            ckpt_path = FLAGS.restore_path.replace('.meta', '')
        else:
            ckpt_path = tf.train.latest_checkpoint(FLAGS.restore_path)
        print('Restoring checkpoint', ckpt_path)
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)
        print('Done')
        
        
        # For testing and validation, there should be only one batch with index 0.
        # im_feats, sent_feats, labels = data_loader.get_batch(0, FLAGS.batch_size, FLAGS.sample_size)
        test_file = 'D:/research/ssnet/voxTrainTestData/faces/facenet_face_veriflist_test_%s_unseenunheard.csv'%(FLAGS.test)
        test_file_voice = 'D:/research/ssnet/voxTrainTestData/voice/voice_veriflist_test_%s_unseenunheard.csv'%(FLAGS.test)
        
            
        
        
        img_test, test_label = read_file_org_test(test_file)
        img_test_voice, _ = read_file_org_test(test_file_voice)
        
        # img_test = img_test[1:]
        # test_label = test_label[1:]
        # img_test = np.asarray(img_test)
        # img_test = img_test[:, :-1]
        
        # img_test_voice = np.asarray(img_test_voice)

        # img_test = img_test[1:]
        # test_label = test_label[1:]
        # img_test_voice = img_test_voice[1:]

        feed_dict = {
            placeholders['im_feat']: img_test,
            placeholders['sent_feat']: img_test_voice,
            placeholders['label']: test_label,
            placeholders['train_phase']: False,
        }
        
        [face, voice] = sess.run([feat_img, feat_sent], feed_dict=feed_dict)
        print(len(face))
        print(len(voice))

        # face = face.tolist()
        # voice =  voice.tolist()

        # exit()

        feat_list = []

        for idx, sfeat in enumerate(face):
            feat_list.append(voice[idx])
            feat_list.append(sfeat)

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
        
        # eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
        print('Equal Error Rate (EER): %1.3f' % eer)
        # plot_tsne(face, voice)
        
        
   

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
    im_feat_dim = 512
    sent_feat_dim = 512

    # Load data.
    # data_loader = DatasetLoader(FLAGS.image_feat_path, FLAGS.sent_feat_path, split='eval')
    # num_ims, im_feat_dim = data_loader.im_feat_shape
    # num_sents, sent_feat_dim = data_loader.sent_feat_shape

    # Setup placeholders for input variables.
    im_feat_plh = tf.placeholder(tf.float32, shape=[None, im_feat_dim])
    sent_feat_plh = tf.placeholder(tf.float32, shape=[None, sent_feat_dim])
    label_plh = tf.placeholder(tf.int64, shape=(None), name='labels')

    train_phase_plh = tf.placeholder(tf.bool)
    placeholders = {
        'im_feat': im_feat_plh,
        'sent_feat': sent_feat_plh,
        'label': label_plh,
        'train_phase': train_phase_plh,
    }

    # Setup testing operation.
    feat1, feat2 = setup_eval_model(im_feat_plh, sent_feat_plh, train_phase_plh, label_plh)

    # Setup checkpoint saver.
    saver = tf.train.Saver(save_relative_paths=True)

    # Periodically evaluate the latest checkpoint in the restore_dir directory,
    # unless a specific chekcpoint MetaGraph path is provided.
    eval_once(saver, placeholders, feat1, feat2)
    # while True:
    #     eval_once(saver, placeholders, feat1, feat2)
    #     if FLAGS.restore_path.endswith('.meta'):
    #         # Only evaluate the given checkpoint.
    #         break
    #     # Set the parameter to match the number of seconds to train 1 epoch.
    #     time.sleep(60)

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)

    parser = argparse.ArgumentParser()
    # Dataset and checkpoints.
    parser.add_argument('--image_feat_path', type=str, help='Path to the image feature mat file.')
    parser.add_argument('--sent_feat_path', type=str, help='Path to the sentence feature mat file.')
    parser.add_argument('--restore_path', type=str, default='./cent_model/',
                        help='Directory for restoring the newest checkpoint or\
                              path to a restoring checkpoint MetaGraph file.')
    parser.add_argument('--test', type=str, default='', help='random, G, N, A, GNA')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
