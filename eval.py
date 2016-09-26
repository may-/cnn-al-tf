# -*- coding: utf-8 -*-

##########################################################
#
# Evaluate Convolutional Neural Network
#
#
#   Note: this implementation is mostly based on
#   https://github.com/yuhaozhang/sentence-convnet/blob/master/eval.py
#
##########################################################

from datetime import datetime
import os
import tensorflow as tf
import numpy as np


import util

FLAGS = tf.app.flags.FLAGS


this_dir = os.path.abspath(os.path.dirname(__file__))

# eval parameters
tf.app.flags.DEFINE_string('train_dir', os.path.join(this_dir, 'train'), 'Directory of the checkpoint files')
tf.app.flags.DEFINE_float('threshold', 0.5, 'Threshold value. Must be one of np.linspace(0, 1, 11)')


def evaluate(eval_data, config):
    """ Build evaluation graph and run. """

    with tf.Graph().as_default():
        with tf.variable_scope('cnn'):
            if config.has_key('split') and config['split']:
                import cnn_split
                m = cnn_split.Model(config, is_train=False)
            else:
                import cnn
                m = cnn.Model(config, is_train=False)
        saver = tf.train.Saver(tf.all_variables())

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(config['train_dir'])
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Loading checkpoint file failed!")

            print "Evaluating ..."
            if config.has_key('split') and config['split']:
                left_batch, middle_batch, right_batch, y_batch, n_batch = zip(*eval_data)
                feed = {m.left: np.array(left_batch), m.right: np.array(right_batch), m.labels: np.array(y_batch)}
            else:
                x_batch, y_batch, n_batch = zip(*eval_data)
                feed = {m.inputs: np.array(x_batch), m.labels: np.array(y_batch)}
            if config['negative']:
                feed[m.negative] = np.array(n_batch)
            loss, eval = sess.run([m.total_loss, m.eval_op], feed_dict=feed)
            pre, rec = zip(*eval)


    return loss, pre, rec



def main(argv=None):
    restore_param = util.load_from_dump(os.path.join(FLAGS.train_dir, 'flags.cPickle'))
    restore_param['train_dir'] = FLAGS.train_dir

    if restore_param.has_key('split') and restore_param['split']:
        data = util.read_data_contextwise(restore_param['data_dir'], 'dev', restore_param['sent_len'],
                                          negative=restore_param['negative'], hierarchical=restore_param['hierarchical'])
    else:
        data = util.read_data(restore_param['data_dir'], 'dev', restore_param['sent_len'],
                              negative=restore_param['negative'], hierarchical=restore_param['hierarchical'])

    loss, pre, rec = evaluate(data, restore_param)
    auc = util.calc_auc_pr(pre, rec)
    try:
        idx = list(np.linspace(0, 1, 11)).index(FLAGS.threshold)
    except ValueError:
        idx = 5
    f1 = (2.0 * pre[idx] * rec[idx]) / (pre[idx] + pre[idx])
    print '%s: loss = %.6f, f1 = %.4f, auc = %.4f' % (datetime.now(), loss, f1, auc)
    util.dump_to_file(os.path.join(FLAGS.train_dir, 'results.cPickle'), {'precision': pre, 'recall': rec})




if __name__ == '__main__':
    tf.app.run()

