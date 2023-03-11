import tensorflow as tf
from tensorflow.contrib.layers.python.layers import fully_connected
# from scipy.spatial import distance
# import numpy as np

hidden_1 = 512
hidden_2 = 256
hidden_3 = 128
face_dim = 4096
voice_dim = 1024
CENTER_LOSS_ALPHA = 0.5


wf = {     
      'h1': tf.Variable(tf.random_normal([face_dim, hidden_1])),
      'h2': tf.Variable(tf.random_normal([hidden_1, hidden_2])),
      'h3': tf.Variable(tf.random_normal([hidden_2, hidden_3])),      
      }

wv = {     
      'h1': tf.Variable(tf.random_normal([voice_dim, hidden_1])),
      'h2': tf.Variable(tf.random_normal([hidden_1, hidden_2])),
      'h3': tf.Variable(tf.random_normal([hidden_2, hidden_3])),      
      }

bf = {
      'b1': tf.Variable(tf.random_normal([hidden_1])),
      'b2': tf.Variable(tf.random_normal([hidden_2])),
      'b3': tf.Variable(tf.random_normal([hidden_3])),
      }
bv = {
      'b1': tf.Variable(tf.random_normal([hidden_1])),
      'b2': tf.Variable(tf.random_normal([hidden_2])),
      'b3': tf.Variable(tf.random_normal([hidden_3])),
      }


def add_fc(inputs, outdim, train_phase, scope_in):
    fc =  fully_connected(inputs, outdim, activation_fn=None, scope=scope_in + '/fc')
    fc_bnorm = tf.layers.batch_normalization(fc, momentum=0.1, epsilon=1e-5,
                         training=train_phase, name=scope_in + '/bnorm')
    fc_relu = tf.nn.relu(fc_bnorm, name=scope_in + '/relu')
    fc_out = tf.layers.dropout(fc_relu, seed=0, training=train_phase, name=scope_in + '/dropout')
    return fc_out

def pdist(x1, x2):
    """
        x1: Tensor of shape (h1, w)
        x2: Tensor of shape (h2, w)
        Return pairwise distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """
    x1_square = tf.reshape(tf.reduce_sum(x1*x1, axis=1), [-1, 1])
    x2_square = tf.reshape(tf.reduce_sum(x2*x2, axis=1), [1, -1])
    return tf.sqrt(x1_square - 2 * tf.matmul(x1, tf.transpose(x2)) + x2_square + 1e-4)

class Wt_Add(tf.keras.layers.Layer):
    def __init__(self, units=1, input_dim=1):
        super(Wt_Add, self).__init__()
        w_init = tf.contrib.layers.xavier_initializer()
        
        self.w1 = tf.Variable(initial_value = w_init(shape = (1,), dtype=tf.float32), trainable = True)
        self.w2 = tf.Variable(initial_value = w_init(shape = (1,), dtype=tf.float32), trainable = True)
        
        # self.w1 = tf.Variable(initial_value = w_init(seed=1, dtype=tf.dtypes.float32), trainable = True)
        # self.w1 = tf.Variable(w_init(1, ))
        # self.w1 = tf.Variable(w_init(shape=(1, ), dtype=np.float32), trainable=True)
        # self.w2 = tf.Variable(initial_value = w_init(shape=(1, ), dtype="float32"), trainable=True)
    
    def call(self, input1, input2):
        return tf.multiply(input1,self.w1) + tf.multiply(input2, self.w2)

def embedding_loss(embeds, labels, FLAGS):
    
    logits = fully_connected(embeds, 901, activation_fn=None)
    
    with tf.variable_scope('loss') as scope:
        
        if FLAGS.loss == 'cent':
            c_loss, _ = center_loss(logits, labels,0.3, 901)
        elif FLAGS.loss == 'git':
            c_loss, _ = git_loss(embeds, labels, 901)
        
        softmax_loss_v = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        
        total_loss = softmax_loss_v + c_loss
    return total_loss

def git_loss(features, labels, num_classes):
    len_features = features.get_shape()[1]
    w_init = tf.contrib.layers.xavier_initializer()
    centers = tf.Variable(initial_value = w_init(shape = (int(num_classes), int(len_features)), dtype=tf.float32), trainable = False)
    
    # centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
    #                           initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels)

    loss = tf.reduce_mean(tf.square(features - centers_batch))

    # Pairwise differences
    diffs = (features[:, tf.newaxis] - centers_batch[tf.newaxis, :])
    diffs_shape = tf.shape(diffs)

    # Mask diagonal (where i == j)
    mask = 1 - tf.eye(diffs_shape[0], diffs_shape[1], dtype=diffs.dtype)
    diffs = diffs * mask[:, :, tf.newaxis]

    # combinaton of two losses
    loss2 = tf.reduce_mean(tf.divide(1, 1 + tf.square(diffs)))

    diff = centers_batch - features
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])

    diff = tf.divide(diff, tf.cast((1 + appear_times), tf.float32))
    diff = CENTER_LOSS_ALPHA * diff

    centers_update_op = tf.scatter_sub(centers, labels, diff)  # diff is used to get updated centers.

    # combo_loss = value_factor * loss + new_factor * loss2
    combo_loss = 1 * loss + 1 * loss2

    return combo_loss, centers_update_op

def center_loss(features, labels, alfa, num_classes):
    nrof_features = features.get_shape()[1]
    # print(num_classes)
    # print(nrof_features)
    w_init = tf.contrib.layers.xavier_initializer()
        
    centers = tf.Variable(initial_value = w_init(shape = (int(num_classes), int(nrof_features)), dtype=tf.float32), trainable = False)
    # centers = tf.Variable('centers', [num_classes, nrof_features], dtype=tf.float32,
    #                           initializer=tf.constant_initializer(0), trainable=False, reuse=tf.AUTO_REUSE)
    label = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers

def recall_k(im_embeds, sent_embeds, im_labels, ks=None):
    """
        Compute recall at given ks.
    """
    sent_im_dist = pdist(sent_embeds, im_embeds)
    def retrieval_recall(dist, labels, k):
        # Use negative distance to find the index of
        # the smallest k elements in each row.
        pred = tf.nn.top_k(-dist, k=k)[1]
        # Create a boolean mask for each column (k value) in pred,
        # s.t. mask[i][j] is 1 iff pred[i][k] = j.
        pred_k_mask = lambda topk_idx: tf.one_hot(topk_idx, labels.shape[1],
                            on_value=True, off_value=False, dtype=tf.bool)
        # Create a boolean mask for the predicted indicies
        # by taking logical or of boolean masks for each column,
        # s.t. mask[i][j] is 1 iff j is in pred[i].
        pred_mask = tf.reduce_any(tf.map_fn(
                pred_k_mask, tf.transpose(pred), dtype=tf.bool), axis=0)
        # Entry (i, j) is matched iff pred_mask[i][j] and labels[i][j] are 1.
        matched = tf.cast(tf.logical_and(pred_mask, labels), dtype=tf.float32)
        return tf.reduce_mean(tf.reduce_max(matched, axis=1))
    return tf.concat(
        [tf.map_fn(lambda k: retrieval_recall(tf.transpose(sent_im_dist), tf.transpose(im_labels), k),
                   ks, dtype=tf.float32),
         tf.map_fn(lambda k: retrieval_recall(sent_im_dist, im_labels, k),
                   ks, dtype=tf.float32)],
        axis=0)


def embedding_model(feats, train_phase, im_labels,
                    fc_dim = 128, embed_dim = 256):
    """
        Build two-branch embedding networks.
        fc_dim: the output dimension of the first fc layer.
        embed_dim: the output dimension of the second fc layer, i.e.
                   embedding space dimension.
    """
    
    
    # Image branch.
    # layer_1 = tf.add(tf.matmul(im_feats, wf['h1']), bf['b1'])
    # layer_2 = tf.add(tf.matmul(layer_1, wf['h2']), bf['b2'])
    # layer_3 = tf.add(tf.matmul(layer_2, wf['h3']), bf['b3'])
    
    fc1 = add_fc(feats, fc_dim, train_phase, 'embed1')
    fc2 = add_fc(fc1, fc_dim, train_phase, 'embed2')
    embeds = tf.nn.l2_normalize(fc2, 1, epsilon=1e-10)
    # im_fc1 = add_fc(im_feats, fc_dim, train_phase, 'im_embed_1')
    
    # im_fc2 = fully_connected(im_fc1, embed_dim, activation_fn=None,
    #                           scope = 'im_embed_2')
    # im_fc2 = tf.layers.dense(im_fc1, embed_dim, activation=tf.nn.tanh)
    # i_embed = tf.nn.l2_normalize(im_fc1, 1, epsilon=1e-10)
    # Voice branch.
    # layer_1 = tf.add(tf.matmul(sent_feats, wv['h1']), bv['b1'])
    # layer_2 = tf.add(tf.matmul(layer_1, wv['h2']), bv['b2'])
    # layer_3 = tf.add(tf.matmul(layer_2, wv['h3']), bv['b3'])
    
    # sent_fc1 = add_fc(sent_feats, fc_dim, train_phase,'sent_embed_1')
    # sent_fc2 = fully_connected(sent_fc1, embed_dim, activation_fn=None,
    #                             scope = 'sent_embed_2')
    # sent_fc2 = tf.layers.dense(sent_fc1, embed_dim, activation=None)
    # s_embed = tf.nn.l2_normalize(sent_fc1, 1, epsilon=1e-10)
    return embeds


def setup_train_model(comb_feats, train_phase, im_labels, FLAGS):
    # im_feats b x image_feature_dim
    # sent_feats 5b x sent_feature_dim
    # train_phase bool (Should be True.)
    # im_labels 5b x b
    embeds = embedding_model(comb_feats, train_phase, im_labels)
    total_loss = embedding_loss(embeds, im_labels, FLAGS)
    return embeds, total_loss

def setup_eval_model(test_feats, train_phase):
    # im_feats b x image_feature_dim
    # sent_feats 5b x sent_feature_dim
    # train_phase bool (Should be False.)
    # im_labels 5b x b
    embeds = embedding_model(test_feats, train_phase, 0)
    #recall = recall_k(i_embed, s_embed, im_labels, ks=tf.convert_to_tensor([1,5,10]))
    return embeds
