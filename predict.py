# -*- coding: utf-8 -*-

##########################################################
#
# Convolutional Neural Network for Multi-label text classification
#
#
#   Note: this implementation is mostly based on
#   https://github.com/yuhaozhang/sentence-convnet/blob/master/eval.py
#
##########################################################

import os
import tensorflow as tf
import numpy as np


import util

FLAGS = tf.app.flags.FLAGS


this_dir = os.path.abspath(os.path.dirname(__file__))

# eval parameters
tf.app.flags.DEFINE_string('train_dir', os.path.join(this_dir, 'train'), 'Directory of the checkpoint files')


def predict(data, config):
    """ Build evaluation graph and run. """

    with tf.Graph().as_default():
        with tf.variable_scope('cnn'):
            if config.has_key('split') and config['split']:
                import cnn_context
                m = cnn_context.Model(config, is_train=False)
            else:
                import cnn
                m = cnn.Model(config, is_train=False)
        saver = tf.train.Saver(tf.all_variables())

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(config['train_dir'])
            saver.restore(sess, ckpt.model_checkpoint_path)

            # predictions
            print "Predicting ..."
            predictions = []
            if config.has_key('split') and config['split']:
                left_batch, right_batch, _, _ = zip(*data)
                feed = {m.left: np.array(left_batch),
                        m.right: np.array(right_batch)}
            else:
                x_batch, _, _ = zip(*data)
                feed = {m.inputs: np.array(x_batch)}
            prob = sess.run(m.scores, feed_dict=feed)
            for i in np.linspace(0, 1, 11):
                prediction = tf.select(prob > i, tf.ones_like(prob), tf.zeros_like(prob))
                predictions.append(prediction.eval())

    # prob.shape = (num_examples, num_classes)
    # predictions.shape = (11, num_examples, num_classes) <- for 11-point PR Curve

    return prob, np.array(predictions)


def embeddings(config):
    """ Build evaluation graph and run. """

    with tf.Graph().as_default():
        saver = tf.train.Saver(tf.all_variables())

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(config['train_dir'])
            saver.restore(sess, ckpt.model_checkpoint_path)

            # embeddings
            embeddings = sess.run(tf.all_variables())

    if config.has_key('split') and config['split']:
        return embeddings[:2]
    else:
        return embeddings[0]

def report(y_true, y_pred, class_names):
    tp = np.logical_and(y_true, y_pred)
    fp = np.logical_and(np.logical_not(y_true), y_pred)
    fn = np.logical_and(y_true, np.logical_not(y_pred))
    pre = np.sum(tp.astype(float), axis=1)/np.sum(np.logical_or(tp, fp).astype(float), axis=1)
    rec = np.sum(tp.astype(float), axis=1)/np.sum(np.logical_or(tp, fn).astype(float), axis=1)
    f1 = (2.0 * pre * rec)/(pre + rec)
    count = np.sum(y_true, axis=0)
    auc_array = []
    ret = '%45s\t P \t  R \t  F1\t AUC\t  C\n' % ' '
    for i, (c, p, r, f, s) in enumerate(zip(class_names, pre[5], rec[5], f1[5], count)):
        auc = util.calc_auc_pr(pre[:, i], rec[:, i])
        auc_array.append(auc)
        ret += '%45s\t%.4f\t%.4f\t%.4f\t%.4f\t%4d\n' % (c, p, r, f, auc, s)

    auc = np.array(auc_array)
    p = pre[5, np.isfinite(pre[5])]*count[np.isfinite(pre[5])]
    r = rec[5, np.isfinite(rec[5])]*count[np.isfinite(rec[5])]
    f = f1[5, np.isfinite(f1[5])]*count[np.isfinite(f1[5])]
    a = auc[np.isfinite(auc)]*count[np.isfinite(auc)]
    ret += '%45s\t%.4f\t%.4f\t%.4f\t%.4f\t%4d\n' % ('total',
                                                    np.sum(p)/count.sum(dtype=float),
                                                    np.sum(r)/count.sum(dtype=float),
                                                    np.sum(f)/count.sum(dtype=float),
                                                    np.sum(a)/count.sum(dtype=float),
                                                    count.sum())

    return ret


def main(argv=None):
    restore_param = util.load_from_dump(os.path.join(FLAGS.train_dir, 'flags.cPickle'))
    restore_param['train_dir'] = FLAGS.train_dir

    if restore_param.has_key('split') and restore_param['split']:
        data = util.read_data_contextwise(restore_param['data_dir'], 'dev', restore_param['sent_len'],
                                          negative=restore_param['negative'])
    else:
        data = util.read_data(restore_param['data_dir'], 'dev', restore_param['sent_len'],
                              negative=restore_param['negative'])

    class_names = util.load_from_dump(os.path.join(restore_param['data_dir'], 'classes.cPickle'))

    y_true = np.array([y[-2] for y in data])
    _, y_pred = predict(data, restore_param)
    print report(y_true, y_pred, class_names)




if __name__ == '__main__':
    tf.app.run()

