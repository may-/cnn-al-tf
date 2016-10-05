# -*- coding: utf-8 -*-

##########################################################
#
# Predict on Convolutional Neural Network
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
tf.app.flags.DEFINE_string('checkpoint_dir', os.path.join(this_dir, 'train'), 'Directory of the checkpoint files')
tf.app.flags.DEFINE_float('threshold', 0.5, 'Threshold value. Must be one of np.linspace(0, 1, 11)')


def predict(data, config):
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
            saver.restore(sess, ckpt.model_checkpoint_path)

            # predictions
            predictions = []
            if config.has_key('split') and config['split']:
                left_batch, right_batch, y_true, _ = zip(*data)
                feed = {m.left: np.array(left_batch), m.right: np.array(right_batch)}
            else:
                x_batch, y_true, _ = zip(*data)
                feed = {m.inputs: np.array(x_batch)}
            prob = sess.run(m.scores, feed_dict=feed)
            for i in np.linspace(0, 1, 11):
                prediction = tf.select(prob > i, tf.ones_like(prob), tf.zeros_like(prob))
                predictions.append(prediction.eval())

    # prob.shape = (num_examples, num_classes)
    # predictions.shape = (11, num_examples, num_classes) <- for 11-point PR Curve

    return prob, np.array(predictions), y_true



def emb(pool, config, class_names=None, relations=None):
    """ Compute gradients with respect tod embeddings layer. """

    config['dropout'] = 0.0 # no dropout
    with tf.Graph().as_default():
        with tf.variable_scope('cnn'):
            if config.has_key('split') and config['split']:
                import cnn_split
                m = cnn_split.Model(config, is_train=True)
            else:
                import cnn
                m = cnn.Model(config, is_train=True)
        saver = tf.train.Saver(tf.all_variables())

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(config['train_dir'])

            learning_rate = 1.0
            if config['optimizer'] == 'adadelta':
                opt = tf.train.AdadeltaOptimizer(learning_rate)
            elif config['optimizer'] == 'adagrad':
                opt = tf.train.AdagradOptimizer(learning_rate)
            elif config['optimizer'] == 'adam':
                opt = tf.train.AdamOptimizer(learning_rate)
            elif config['optimizer'] == 'sgd':
                opt = tf.train.GradientDescentOptimizer(learning_rate)
            else:
                raise ValueError("Optimizer not supported.")

            m.assign_lr(sess, config['init_lr'])
            gradients = []

            # feed only one instance at a time (batch_size = 1)
            for instance in pool:
                print '.',
                saver.restore(sess, ckpt.model_checkpoint_path)
                if config.has_key('split') and config['split']:
                    # get prediction
                    feed = {m.left: instance[0].reshape((1, config['sent_len'])),
                            m.right: instance[1].reshape((1, config['sent_len']))}
                    prob = sess.run(m.scores, feed_dict=feed)

                    # get total loss
                    label = [0.0] * config['num_classes']
                    losses = []
                    for i in range(config['num_classes']):
                        label[i] = 1.0
                        feed[m.labels] = np.array(label).reshape((1, config['num_classes']))
                        if config['negative'] and class_names and relations:
                            neg = util.pseudo_negative_sampling(np.array(label), class_names, relations,
                                                                hierarchical=config['hierarchical'])
                            feed[m.negative] = np.array(neg.eval()).reshape((1, config['num_classes']))
                        _, loss = sess.run([m.train_op, m.total_loss], feed_dict=feed)
                        losses.append(loss)

                else:
                    # get probability
                    feed = {m.inputs: instance[0].reshape((1, config['sent_len']))}
                    prob = sess.run(m.scores, feed_dict=feed)

                    # get total loss
                    label = [0.0] * config['num_classes']
                    losses = []
                    for i in range(config['num_classes']):
                        label[i] = 1.0
                        feed[m.labels] = np.array(label).reshape((1, config['num_classes']))
                        if config['negative'] and class_names and relations:
                            neg = util.pseudo_negative_sampling(np.array(label), class_names, relations,
                                                                hierarchical=config['hierarchical'])
                            feed[m.negative] = np.array(neg).reshape((1, config['num_classes']))
                        _, loss = sess.run([m.train_op, m.total_loss], feed_dict=feed)
                        losses.append(loss)

                # get variable by name
                emb = [var for var in tf.trainable_variables() if var.op.name.startswith('cnn/embedding')]

                marginal = []
                scaled = util.minmax_scale(prob[0, :])
                assert len(scaled) == config['num_classes']
                for i, loss in enumerate(losses):
                    # compute gradients w.r.t. embeddings layer
                    grad = opt.compute_gradients(tf.cast(loss, dtype=tf.float32), emb)
                    # compute norm and scale by the probability
                    marginal.append(scaled[i] * np.linalg.norm(np.array([g[1].eval() for g in grad])))
                gradients.append(np.sum(marginal))

    return gradients



def report(y_true, y_pred, class_names, threshold=0.5):
    tp = np.logical_and(y_true, y_pred)
    fp = np.logical_and(np.logical_not(y_true), y_pred)
    fn = np.logical_and(y_true, np.logical_not(y_pred))
    pre = np.sum(tp.astype(float), axis=1)/np.sum(np.logical_or(tp, fp).astype(float), axis=1)
    rec = np.sum(tp.astype(float), axis=1)/np.sum(np.logical_or(tp, fn).astype(float), axis=1)
    f1 = (2.0 * pre * rec)/(pre + rec)
    count = np.sum(y_true, axis=0)
    auc_array = []
    ret = '%45s\t   P\t   R\t  F1\t  AUC\t  C\n' % ' '
    for i, (c, p, r, f, s) in enumerate(zip(class_names, pre[5], rec[5], f1[5], count)):
        auc = util.calc_auc_pr(pre[:, i], rec[:, i])
        auc_array.append(auc)
        ret += '%45s\t%.4f\t%.4f\t%.4f\t%.4f\t%4d\n' % (c, p, r, f, auc, s)

    auc = np.array(auc_array)
    try:
        idx = list(np.linspace(0, 1, 11)).index(FLAGS.threshold)
    except ValueError:
        idx = 5
    p = pre[idx, np.isfinite(pre[idx])]*count[np.isfinite(pre[idx])]
    r = rec[idx, np.isfinite(rec[idx])]*count[np.isfinite(rec[idx])]
    f = f1[idx, np.isfinite(f1[idx])]*count[np.isfinite(f1[idx])]
    a = auc[np.isfinite(auc)]*count[np.isfinite(auc)]
    ret += '%45s\t%.4f\t%.4f\t%.4f\t%.4f\t%4d\n' % ('avg.',
                                                    np.sum(p)/count.sum(dtype=float),
                                                    np.sum(r)/count.sum(dtype=float),
                                                    np.sum(f)/count.sum(dtype=float),
                                                    np.sum(a)/count.sum(dtype=float),
                                                    count.sum())

    return ret


def main(argv=None):
    restore_param = util.load_from_dump(os.path.join(FLAGS.checkpoint_dir, 'flags.cPickle'))
    restore_param['train_dir'] = FLAGS.checkpoint_dir

    if restore_param.has_key('split') and restore_param['split']:
        data = util.read_data_contextwise(restore_param['data_dir'], 'dev', restore_param['sent_len'],
                                          negative=restore_param['negative'], hierarchical=restore_param['hierarchical'])
    else:
        data = util.read_data(restore_param['data_dir'], 'dev', restore_param['sent_len'],
                              negative=restore_param['negative'], hierarchical=restore_param['hierarchical'])

    class_names = util.load_from_dump(os.path.join(restore_param['data_dir'], 'classes.cPickle'))

    print "Predicting ..."
    _, y_pred, y_true = predict(data, restore_param)
    print "Performance (threshold=%.1f)" % FLAGS.threshold
    print report(y_true, y_pred, class_names, threshold=FLAGS.threshold)




if __name__ == '__main__':
    tf.app.run()

