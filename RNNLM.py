import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
import math

class RNNLM(object):
    def __init__(self,
                 vocab_size,
                 batch_size,
                 num_epochs,
                 check_point_step,
                 num_train_samples,
                 num_valid_samples,
                 num_layers,
                 num_hidden_units,
                 max_gradient_norm,
                 initial_learning_rate=1,
                 final_learning_rate=0.001
                 ):

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.check_point_step = check_point_step
        self.num_train_samples = num_train_samples
        self.num_valid_samples = num_valid_samples
        self.num_layers = num_layers
        self.num_hidden_units = num_hidden_units
        self.max_gradient_norm = max_gradient_norm

        self.global_step = tf.Variable(0, trainable=False)

        # We set a dynamic learining rate, it decays every time the model has gone through 150 batches.
        # A minimum learning rate has also been set.
        self.learning_rate = tf.train.exponential_decay(initial_learning_rate, self.global_step, 150, 0.96, staircase=True)
        self.learning_rate = tf.cond(tf.less(self.learning_rate, final_learning_rate), lambda: tf.constant(final_learning_rate), lambda: self.learning_rate)

        self.dropout_rate = tf.placeholder(tf.float32, name="dropout_rate")

        self.file_name_train = tf.placeholder(tf.string)
        self.file_name_validation = tf.placeholder(tf.string)
        self.file_name_test = tf.placeholder(tf.string)

        def parse(line):
            '''
            padded_batch(
                batch_size,
                padded_shapes,        # looks like we can do something with this actually. force a certain size ?
                padding_values=None,
                drop_remainder=False)
            '''

            line_split = tf.string_split([line])

            input_seq = tf.string_to_number(line_split.values[:-1], out_type=tf.int32)
            output_seq = tf.string_to_number(line_split.values[1:], out_type=tf.int32)

            input_seq = input_seq[0:64]
            output_seq = output_seq[0:64]

            shape = tf.shape(input_seq)

            input_seq = tf.pad(input_seq,   [[0, 64 - shape[0]]])
            output_seq = tf.pad(output_seq, [[0, 64 - shape[0]]])

            # input_seq = tf.Print(input_seq, [tf.shape(input_seq), tf.shape(output_seq)], message='', summarize=1000)
            # input_seq = tf.Print(input_seq, [tf.reduce_max(input_seq), tf.reduce_max(output_seq)], message='', summarize=1000)

            return input_seq, output_seq
            
        ######################################

        training_dataset = tf.data.TextLineDataset(self.file_name_train).map(parse).batch(self.batch_size, drop_remainder=True).repeat()
        validation_dataset = tf.data.TextLineDataset(self.file_name_validation).map(parse).batch(self.batch_size, drop_remainder=True).repeat()
        test_dataset = tf.data.TextLineDataset(self.file_name_test).map(parse).batch(1)

        iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)

        self.input_batch, self.output_batch = iterator.get_next()

        self.trining_init_op = iterator.make_initializer(training_dataset)
        self.validation_init_op = iterator.make_initializer(validation_dataset)
        self.test_init_op = iterator.make_initializer(test_dataset)

        ######################################

        non_zero_weights = tf.sign(self.input_batch)
        self.valid_words = tf.reduce_sum(non_zero_weights)

        labels = tf.reshape(self.output_batch, [-1])

        # Input embedding mat
        self.input_embedding_mat = tf.get_variable("input_embedding_mat", [self.vocab_size, self.num_hidden_units], dtype=tf.float32)
        self.input_embedded = tf.nn.embedding_lookup(self.input_embedding_mat, self.input_batch)

        with tf.variable_scope('l1'):
            cell = tf.contrib.rnn.LSTMCell(self.num_hidden_units, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropout_rate)
            self.cell1 = cell
            self.outputs1, _ = tf.nn.dynamic_rnn(cell=self.cell1, inputs=self.input_embedded, dtype=tf.float32)
            self.output_embedding_mat1 = tf.get_variable("output_embedding_mat1", [self.vocab_size, self.num_hidden_units], dtype=tf.float32)
            self.output_embedding_bias1 = tf.get_variable("output_embedding_bias1", [self.vocab_size], dtype=tf.float32)
            logits1 = tf.matmul(self.outputs1, tf.transpose(self.output_embedding_mat1)) + self.output_embedding_bias1
            self.logits1 = tf.reshape(logits1, [-1, vocab_size])
            self.loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.logits1) * tf.cast(tf.reshape(non_zero_weights, [-1]), tf.float32)
            self.params1 = tf.trainable_variables('l1')
            gradients1 = tf.gradients(self.loss1, self.params1, colocate_gradients_with_ops=True)
            self.gradients1, _ = tf.clip_by_global_norm(gradients1, self.max_gradient_norm)
            

        with tf.variable_scope('l2'):
            cell = tf.contrib.rnn.LSTMCell(self.num_hidden_units, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropout_rate)
            self.cell2 = cell
            self.outputs2, _ = tf.nn.dynamic_rnn(cell=self.cell2, inputs=self.outputs1, dtype=tf.float32)
            self.output_embedding_mat2 = tf.get_variable("output_embedding_mat2", [self.vocab_size, self.num_hidden_units], dtype=tf.float32)
            self.output_embedding_bias2 = tf.get_variable("output_embedding_bias2", [self.vocab_size], dtype=tf.float32)
            logits2 = tf.matmul(self.outputs2, tf.transpose(self.output_embedding_mat2)) + self.output_embedding_bias2
            self.logits2 = tf.reshape(logits2, [-1, vocab_size])
            self.loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.logits2) * tf.cast(tf.reshape(non_zero_weights, [-1]), tf.float32)
            self.params2 = tf.trainable_variables('l2')
            gradients2 = tf.gradients(self.loss2, self.params2, colocate_gradients_with_ops=True)
            self.gradients2, _ = tf.clip_by_global_norm(gradients2, self.max_gradient_norm)
            
        ######################################

        self.gradients = self.gradients1 + self.gradients2
        self.params = self.params1 + self.params2
        self.train = tf.train.AdagradOptimizer(self.learning_rate).apply_gradients(zip(self.gradients, self.params), global_step=self.global_step)
        self.loss = self.loss2

    def batch_train(self, sess, saver):

        for epoch in range(self.num_epochs):

            ##############################

            sess.run(self.trining_init_op, {self.file_name_train: "./data/train.ids"})
            train_loss = 0.0
            train_valid_words = 0
            
            for ii in range(0, self.num_train_samples, self.batch_size):
                # print ('%d / %d' % (ii, self.num_train_samples))               

                _loss, _valid_words, global_step, current_learning_rate, _, _params, grads = sess.run([self.loss, self.valid_words, self.global_step, self.learning_rate, self.train, self.params, self.gradients], {self.dropout_rate: 1.0})

                '''
                for p in _params:
                    print (np.shape(p), np.std(p), np.average(p))
                assert(False)
                '''
                '''
                for g in grads:
                    if np.shape(g) == (3,):
                        print (np.shape(g[0]), np.std(g[0]), np.average(g[0]))
                    else:
                        print (np.shape(g), np.std(g), np.average(g))
                assert(False)
                '''

                train_loss += np.sum(_loss)
                train_valid_words += _valid_words

                if global_step % self.check_point_step == 0:

                    train_loss /= train_valid_words
                    train_ppl = math.exp(train_loss)
                    print ("step: %d, lr: %f, ppl: %f" % (global_step, current_learning_rate, train_ppl))

                    train_loss = 0.0
                    train_valid_words = 0
                    
            ##############################

            sess.run(self.validation_init_op, {self.file_name_validation: "./data/valid.ids"})
            dev_loss = 0.0
            dev_valid_words = 0
            
            for _ in range(0, self.num_valid_samples, self.batch_size):

                _dev_loss, _dev_valid_words = sess.run([self.loss, self.valid_words], {self.dropout_rate: 1.0})
                dev_loss += np.sum(_dev_loss)
                dev_valid_words += _dev_valid_words

            dev_loss /= dev_valid_words
            dev_ppl = math.exp(dev_loss)
            print ("val ppl: %f" % (dev_ppl))
            
            ##############################

    def predict(self, sess, input_file, raw_file, verbose=False):
        # if verbose is true, then we print the ppl of every sequence

        sess.run(self.test_init_op, {self.file_name_test: input_file})

        with open(raw_file) as fp:

            global_dev_loss = 0.0
            global_dev_valid_words = 0

            for raw_line in fp.readlines():

                raw_line = raw_line.strip()

                _dev_loss, _dev_valid_words, input_line = sess.run([self.loss, self.valid_words, self.input_batch], {self.dropout_rate: 1.0})

                dev_loss = np.sum(_dev_loss)
                dev_valid_words = _dev_valid_words

                global_dev_loss += dev_loss
                global_dev_valid_words += dev_valid_words

                if verbose:
                    dev_loss /= dev_valid_words
                    dev_ppl = math.exp(dev_loss)
                    print (raw_line + "    Test PPL: {}".format(dev_ppl))

            global_dev_loss /= global_dev_valid_words
            global_dev_ppl = math.exp(global_dev_loss)
            print ("Global Test PPL: {}".format(global_dev_ppl))
            
            
