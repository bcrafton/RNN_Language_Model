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
            line_split = tf.string_split([line])
            input_seq = tf.string_to_number(line_split.values[:-1], out_type=tf.int32)
            output_seq = tf.string_to_number(line_split.values[1:], out_type=tf.int32)

            # line = tf.Print(line, [self.batch_size, line], message='', summarize=1000)
            
            # line = tf.Print(line, [input_seq], message='', summarize=1000)
            # line = tf.Print(line, [output_seq], message='', summarize=1000)

            return input_seq, output_seq

        training_dataset = tf.data.TextLineDataset(self.file_name_train).map(parse).padded_batch(self.batch_size, padded_shapes=([None], [None]))
        validation_dataset = tf.data.TextLineDataset(self.file_name_validation).map(parse).padded_batch(self.batch_size, padded_shapes=([None], [None]))
        
        # training_dataset = tf.data.TextLineDataset(self.file_name_train).map(parse).batch(1)
        # validation_dataset = tf.data.TextLineDataset(self.file_name_validation).map(parse).batch(1)
        test_dataset = tf.data.TextLineDataset(self.file_name_test).map(parse).batch(1)

        iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)

        self.input_batch, self.output_batch = iterator.get_next()

        self.trining_init_op = iterator.make_initializer(training_dataset)
        self.validation_init_op = iterator.make_initializer(validation_dataset)
        self.test_init_op = iterator.make_initializer(test_dataset)

        # Input embedding mat
        self.input_embedding_mat = tf.get_variable("input_embedding_mat", [self.vocab_size, self.num_hidden_units], dtype=tf.float32)
        self.input_embedding_bias = tf.zeros(shape=(self.batch_size, 25, self.num_hidden_units))
        self.input_embedded = tf.nn.embedding_lookup(self.input_embedding_mat, self.input_batch) # + self.input_embedding_bias
        # self.input_embedded = tf.Print(self.input_embedded, [tf.shape(self.input_embedded)], message='', summarize=1000)

        # LSTM cell
        cell = tf.contrib.rnn.LSTMCell(self.num_hidden_units, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropout_rate)
        cell = tf.contrib.rnn.MultiRNNCell(cells=[cell]*self.num_layers, state_is_tuple=True)
        self.cell = cell

        # Output embedding
        self.output_embedding_mat = tf.get_variable("output_embedding_mat", [self.vocab_size, self.num_hidden_units], dtype=tf.float32)
        self.output_embedding_bias = tf.get_variable("output_embedding_bias", [self.vocab_size], dtype=tf.float32)

        non_zero_weights = tf.sign(self.input_batch)
        self.valid_words = tf.reduce_sum(non_zero_weights)

        # Compute sequence length
        def get_length(non_zero_place):
            real_length = tf.reduce_sum(non_zero_place, 1)
            real_length = tf.cast(real_length, tf.int32)
            return real_length

        batch_length = get_length(non_zero_weights)
        # batch_length = tf.Print(batch_length, [batch_length], message='', summarize=1000)

        # The shape of outputs is [batch_size, max_length, num_hidden_units]
        outputs, _ = tf.nn.dynamic_rnn(cell=self.cell, inputs=self.input_embedded, sequence_length=batch_length, dtype=tf.float32)

        def output_embedding(current_output):
            return tf.add(tf.matmul(current_output, tf.transpose(self.output_embedding_mat)), self.output_embedding_bias)

        # To compute the logits
        labels = tf.reshape(self.output_batch, [-1])

        logits = tf.map_fn(output_embedding, outputs)
        logits = tf.reshape(logits, [-1, vocab_size])

        # logits = tf.Print(logits, [tf.shape(logits), tf.shape(labels)], message='', summarize=1000)

        # because of how many 0==0 we will have, this makes me question our perplexity score here.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits) * tf.cast(tf.reshape(non_zero_weights, [-1]), tf.float32)
        self.loss = loss

        # Train
        params = tf.trainable_variables()

        opt = tf.train.AdagradOptimizer(self.learning_rate)
        gradients = tf.gradients(self.loss, params, colocate_gradients_with_ops=True)
        
        # [self.lu_grad] = tf.gradients(self.loss, [self.input_embedding_mat], colocate_gradients_with_ops=True)
        # [self.lu_bias_grad] = tf.gradients(self.loss, [self.input_embedding_bias], colocate_gradients_with_ops=True)        

        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.updates = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)


    def batch_train(self, sess, saver):

        best_score = np.inf
        patience = 5
        epoch = 0

        while epoch < self.num_epochs:

            sess.run(self.trining_init_op, {self.file_name_train: "./data/train.ids"})
            train_loss = 0.0
            train_valid_words = 0
            while True:

                # _loss, _valid_words, _, i, o, grad1, grad2 = sess.run([self.loss, self.valid_words, self.updates, self.input_batch, self.output_batch, self.lu_grad, self.lu_bias_grad], {self.dropout_rate: 0.5})
                _loss, _valid_words, _, global_step, i, o = sess.run([self.loss, self.valid_words, self.updates, self.global_step, self.input_batch, self.output_batch], {self.dropout_rate: 0.5})
                '''
                print (np.shape(grad1[0]), np.shape(grad1[1]), np.shape(grad1[2]))
                print (np.shape(grad2))

                print (grad2[0][0]) # [batch]    [time][hidden]
                print (grad1[0][0]) # [grad comp][time][hidden]
                '''
                '''
                print (np.shape(grad1))
                print (np.shape(grad1[0]), np.shape(grad1[1]), np.shape(grad1[2]))
                print (grad1[1][0], grad1[1][1], grad1[1][2])
                print (grad1[2])
                '''

                # print (len(i))
                # print (len(o))                    

                for sentence in i:
                    pass
                    # print (np.count_nonzero(sentence == 0), np.count_nonzero(sentence), len(sentence))

                count = 0
                for sentence in i:
                    count += len(sentence)
                # print (count)

                count = 0
                for sentence in o:
                    count += len(sentence)
                # print (count)

                # print (i[0])
                # print (o[0])

                # print (np.shape(i))
                # print (np.shape(o))

                # assert (False)

                train_loss += np.sum(_loss)
                train_valid_words += _valid_words

                if global_step % self.check_point_step == 0:

                    train_loss /= train_valid_words
                    train_ppl = math.exp(train_loss)
                    # print ("Training Step: {}, LR: {}".format(global_step, current_learning_rate))
                    print ("    Training PPL: {}".format(train_ppl))

                    train_loss = 0.0
                    train_valid_words = 0


            sess.run(self.validation_init_op, {self.file_name_validation: "./data/valid.ids"})
            dev_loss = 0.0
            dev_valid_words = 0
            while True:
                try:
                    _dev_loss, _dev_valid_words = sess.run(
                        [self.loss, self.valid_words],
                        {self.dropout_rate: 1.0})

                    dev_loss += np.sum(_dev_loss)
                    dev_valid_words += _dev_valid_words

                except tf.errors.OutOfRangeError:
                    dev_loss /= dev_valid_words
                    dev_ppl = math.exp(dev_loss)
                    print ("Validation PPL: {}".format(dev_ppl))
                    if dev_ppl < best_score:
                        patience = 5
                        saver.save(sess, "model/best_model.ckpt")
                        best_score = dev_ppl
                    else:
                        patience -= 1

                    if patience == 0:
                        epoch = self.num_epochs

                    break

    def predict(self, sess, input_file, raw_file, verbose=False):
        # if verbose is true, then we print the ppl of every sequence

        sess.run(self.test_init_op, {self.file_name_test: input_file})

        with open(raw_file) as fp:

            global_dev_loss = 0.0
            global_dev_valid_words = 0

            for raw_line in fp.readlines():

                raw_line = raw_line.strip()

                _dev_loss, _dev_valid_words, input_line = sess.run(
                    [self.loss, self.valid_words, self.input_batch],
                    {self.dropout_rate: 1.0})
 
                assert(False)

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

