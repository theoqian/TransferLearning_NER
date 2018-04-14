import tensorflow as tf
from tensorflow.contrib.layers.python.layers.layers import initializers
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.crf import crf_log_likelihood, viterbi_decode
import numpy as np


class SharedModel(object):
    """The shared part of different NER models.
    """

    reuse = False

    def __init__(self,
                 num_chars,
                 char_dim,
                 lstm_dim,
                 initializer,
                 name='shared_model'):
        self._num_chars = num_chars
        self._char_dim = char_dim
        self._lstm_dim = lstm_dim
        self._initializer = initializer
        with tf.variable_scope(name, reuse=SharedModel.reuse):
            self.char_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='char_inputs')
            self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name='targets')
            self.dropout_rate = tf.placeholder(dtype=tf.float32, name='dropout_rate')
            self.lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='input_length')
            embedding = self._embedding_layer(self.char_inputs)
            lstm_inputs = tf.nn.dropout(embedding, self.dropout_rate)
            self._lstm_outputs = self._bilstm_layer(lstm_inputs, self._lstm_dim, self.lengths)
        SharedModel.reuse = True

    def _embedding_layer(self, char_inputs, name='embedding_layer'):
        with tf.variable_scope(name), tf.device('/cpu:0'):
            self.char_lookup = tf.get_variable(name='char_embedding',
                                               shape=[self._num_chars, self._char_dim],
                                               initializer=self._initializer,
                                               trainable=False)
            embedding = tf.nn.embedding_lookup(self.char_lookup, char_inputs)
        return embedding

    def _bilstm_layer(self, lstm_inputs, lstm_dim, lengths, name='BiLSTM_layer'):
        with tf.variable_scope(name):
            cell_fw = LSTMCell(lstm_dim)
            cell_bw = LSTMCell(lstm_dim)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths
            )
        return tf.concat([output_fw, output_bw], axis=-1)

    def output(self):
        return self._lstm_outputs


class TransformModel(object):
    """Transform model
    """

    def __init__(self,
                 num_chars,
                 num_target,
                 char_dim,
                 lstm_dim,
                 name,
                 init_lr=0.001,
                 grad_clip=5,
                 project_dim=None):
        project_dim = project_dim if project_dim else lstm_dim
        self._num_chars = num_chars
        self._num_targets = num_target
        self._char_dim = char_dim
        self._lstm_dim = lstm_dim
        self._global_step = tf.Variable(0, trainable=False)
        self._initializer = initializers.xavier_initializer()
        self._shared_layers = SharedModel(num_chars=self._num_chars,
                                          char_dim=self._char_dim,
                                          lstm_dim=self._lstm_dim,
                                          initializer=self._initializer)
        self._char_inputs = self._shared_layers.char_inputs
        self._targets = self._shared_layers.targets
        self._dropout_rate = self._shared_layers.dropout_rate
        self._lengths = self._shared_layers.lengths
        self._batch_size = tf.shape(self._char_inputs)[0]
        self._num_steps = tf.shape(self._char_inputs)[-1]
        self._logits = self._project_layer(self._shared_layers.output(),
                                           project_dim,
                                           name + '_project')
        self._loss = self._crf_loss(self._logits, self._lengths, name + "_loss")

        with tf.variable_scope(name + "optimizer"):
            self.optimizer = tf.train.AdamOptimizer(init_lr)
            #grads_vars = self.optimizer.compute_gradients(self._loss)
            #capped_grads_var = [[tf.clip_by_value(g, -grad_clip, grad_clip), v] for g, v in grads_vars]
            #self._train_op = self.optimizer.apply_gradients(capped_grads_var, self._global_step)
            self._train_op = self.optimizer.minimize(self._loss, self._global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def _project_layer(self, lstm_outputs, hidden_dim, name):
        with tf.variable_scope(name):
            with tf.variable_scope('hidden'):
                w = tf.get_variable('w',
                                    shape=[self._lstm_dim * 2, hidden_dim],
                                    dtype=tf.float32,
                                    initializer=self._initializer)
                b = tf.get_variable('b',
                                    shape=[hidden_dim],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self._lstm_dim * 2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, w, b))

            with tf.variable_scope('logits'):
                w = tf.get_variable('w',
                                    shape=[hidden_dim, self._num_targets],
                                    dtype=tf.float32,
                                    initializer=self._initializer)
                b = tf.get_variable('b',
                                    shape=[self._num_targets],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                pred = tf.nn.xw_plus_b(hidden, w, b)

            return tf.reshape(pred, [-1, self._num_steps, self._num_targets])

    def _crf_loss(self, logits, lengths, name):
        with tf.variable_scope(name):
            small = -1000.0
            start_logits = tf.concat(
                [small * tf.ones(shape=[self._batch_size, 1, self._num_targets + 1]),
                 tf.zeros(shape=[self._batch_size, 1, 1])],
                axis=-1
            )
            end_logits = tf.concat(
                [small * tf.ones(shape=[self._batch_size, 1, self._num_targets]),
                 tf.zeros(shape=[self._batch_size, 1, 1]),
                 small * tf.ones(shape=[self._batch_size, 1, 1])],
                axis=-1
            )
            pad_logits = tf.cast(small * tf.ones([self._batch_size, self._num_steps, 2]), tf.float32)
            logits = tf.concat([logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits, end_logits], axis=1)
            targets = tf.concat(
                [tf.cast((self._num_targets + 1) * tf.ones([self._batch_size, 1]), tf.int32),
                 self._targets,
                 tf.cast(self._num_targets * tf.ones([self._batch_size, 1]), tf.int32)],
                axis=-1
            )
            log_likelihood, self._transition = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                sequence_lengths=lengths + 2
            )
            return -tf.reduce_mean(log_likelihood)

    def _create_feed_dict(self, is_train, batch):
        _, chars, tags, lengths = batch
        feed_dict = {
            self._char_inputs: np.asarray(chars),
            self._lengths: np.asarray(lengths),
            self._dropout_rate: 1.0
        }
        if is_train:
            feed_dict[self._targets] = np.asarray(tags)
            feed_dict[self._dropout_rate] = 0.5
        return feed_dict

    def char_lookup(self):
        return self._shared_layers.char_lookup

    def run_step(self, sess, is_train, batch):
        feed_dict = self._create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run([self._global_step, self._loss, self._train_op],
                                            feed_dict)
            return global_step, loss
        else:
            logits = sess.run([self._logits],
                              feed_dict)
            return logits[0]

    def decode(self, logits, lengths, matrix):
        paths = []
        small = -1000.0
        start = np.asarray([[small] * self._num_targets + [small] + [0]])
        end = np.asarray([[small] * self._num_targets + [0] + [small]])
        for logit, length in zip(logits, lengths):
            logit = logit[:length]
            pad = small * np.ones([length, 2])
            logit = np.concatenate([logit, pad], axis=1)
            logit = np.concatenate([start, logit, end], axis=0)
            path, _ = viterbi_decode(logit, matrix)
            paths.append(path[1:-1])
        return paths

    def evaluate_line(self, sess, inputs, id2tag):
        lengths = inputs[-1]
        trans = self._transition.eval()
        logits = self.run_step(sess, False, inputs)
        paths = self.decode(logits, lengths, trans)
        tags = [id2tag[idx] for idx in paths[0]]
        return tags

    def evaluate(self, sess, data_manger, id2tag):
        results = []
        trans = self._transition.eval()
        for batch in data_manger.iter_batch():
            strings, chars, tags, lengths = batch
            logits = self.run_step(sess, False, batch)
            paths = self.decode(logits, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                lab = [id2tag[x] for x in tags[i][:lengths[i]]]
                pred = [id2tag[x] for x in paths[i][:lengths[i]]]
                for char, lab, pred in zip(string, lab, pred):
                    result.append(" ".join([char, lab, pred]))
                results.append(result)
        return results
