import tensorflow as tf

import modeling as bert_modeling
import tokenization
from base_model import BaseModel
from bert_input import InputExample, InputFeatures
from utils import process_bert_embeddings

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Config(object):

    def __init__(self):
        self.max_length = 40 # longest sequence to parse
        self.n_classes = 10
        self.dropout = 0.5
        self.hidden_size = 300
        self.n_epochs = 10
        self.filter_sizes = [3, 4, 5]
        self.filter_nums = 2
        #self.max_grad_norm = 10.
        self.lr = 0.001


def construct_vocab(path):
    with open(path, encoding='utf-8') as f:
        vocab = set([x.strip() for x in f.readlines() if x.strip()])
    return vocab

class textCNN(BaseModel):
    def __init__(self, max_len,  batch_size, n_epochs, n_classes = None):
        super(textCNN, self).__init__(Config())
        self.config.max_length = max_len
        self.config.n_epochs = n_epochs
        if n_classes != None:
            self.config.n_classes = n_classes
        self.build()

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for line in lines:
            text_a = tokenization.convert_to_unicode(line)

            examples.append(
                InputExample(text_a=text_a))
        # print(lines)

        # sys.exit(examples)
        return examples

    def convert_examples_to_features(self, examples, tokenizer):
        """Convert a set of `InputExample`s to a list of `InputFeatures`."""

        features = []
        for example in examples:
            feature = self.convert_single_example(example, tokenizer)

            features.append(feature)
            
        return features

    def convert_single_example(self, example, tokenizer):
        """Converts a single `InputExample` into a single `InputFeatures`."""
        # tokens_a = tokenizer.tokenize(example.text_a)  ###
        tokens_a = [x if x in self.vocab else '[UNK]' for x in list(example.text_a)]

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > self.config.max_length - 2:
            tokens_a = tokens_a[0:(self.config.max_length - 2)]
        tokens = ["[CLS]"]

        for token in tokens_a:
            tokens.append(token)

        tokens.append("[SEP]")
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # print (input_ids)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.config.max_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == self.config.max_length
        assert len(input_mask) == self.config.max_length

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask
        )
        return feature

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None, None, 768], name = 'embedding')
        self.label_placeholder = tf.placeholder(tf.float32, [None, self.config.n_classes])
        self.dropout_placeholder = tf.placeholder(tf.float64)

    def create_feed_dict(self, inputs_batch, labels_batch=None,  dropout_prob=1.):
        sequence_lengths = [len(x) for x in inputs_batch]
        bert_embeddings = self.encode_bert(inputs_batch)

        embeddings_padded = process_bert_embeddings(bert_embeddings, sequence_lengths, self.config.max_length)

        feed_dict = {
            self.input_placeholder : embeddings_padded,
            self.dropout_placeholder : dropout_prob
        }
        if labels_batch is not None:
            feed_dict[self.label_placeholder] = labels_batch

        return feed_dict


    def add_conv_op(self):
        with tf.variable_scope('conv_pool'):
            conv_pools = []
            for filter_size in self.config.filter_sizes:
                with tf.variable_scope('conv_pool_{}'.format(filter_size)):
                    conv = tf.layers.conv1d(
                        self.input_placeholder, 
                        filters = self.config.filter_nums,
                        kernel_size = filter_size,
                        strides = 1,
                        padding = 'valid',
                        activation = tf.nn.relu,
                        name = 'conv_{}'.format(filter_size))

                    conv_pool = tf.layers.max_pooling1d(
                        conv,
                        pool_size = self.config.max_length - filter_size + 1,
                        strides = 1,
                        padding='valid',
                        name = 'max_pool_{}'.format(filter_size)
                    )
                    conv_pools.append(conv_pool)
            
            hidden = tf.concat(conv_pools, -1)
            hidden = tf.reshape(hidden, [-1, self.config.filter_nums * len(self.config.filter_sizes)], name='pool_flat')
            return hidden

    def add_prediction_op(self):
        with tf.name_scope("dropout"):
            hidden_drop = tf.nn.dropout(self.hidden, self.dropout_placeholder)

        with tf.variable_scope('dense'):
            hidden = tf.layers.dense(hidden_drop, self.config.hidden_size, activation=tf.nn.relu, name = 'hidden')
            logits = tf.layers.dense(hidden, self.config.n_classes, name = 'logits')
            pred = tf.argmax(tf.nn.softmax(logits), 1)
            return logits, pred

    def add_loss_op(self):
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits, labels = self.label_placeholder), name= 'loss')
            tf.summary.scalar("loss", loss)
            return loss

    def add_accuracy_op(self):
        with tf.variable_scope('accuracy'):
            correct_predictions = tf.equal(self.pred, tf.argmax(self.label_placeholder, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name = 'accuracy')
            return accuracy

    def add_F1_op(self):
        with tf.variable_scope('F1'):
            epsilon = 1e-7
            pred_hat = tf.one_hot(indices = self.pred, depth = self.config.n_classes,axis = 1)
            tp = tf.reduce_sum(tf.cast(pred_hat * self.label_placeholder, 'float'), axis = 0, name='tp')
            fp = tf.reduce_sum(tf.cast(pred_hat * (1 - self.label_placeholder), 'float'), axis = 0, name='fp')
            fn = tf.reduce_sum(tf.cast((1 - pred_hat) * self.label_placeholder, 'float'), axis = 0, name='fn')
            
            p = tp / (tp + fn + epsilon)
            r = tp / (tp + fp + epsilon)

            f1 = tf.reduce_mean(2 * p * r / (p + r + epsilon), name = 'F1_score')
            return f1

    def add_training_op(self):
        train_op = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss)
        return train_op

    def train_on_batch(self, inputs_batch, labels_batch, dropout_prob = 1.):
        feed = self.create_feed_dict(inputs_batch, labels_batch, dropout_prob)
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def board_on_batch(self, inputs_batch, labels_batch, index, dropout_prob = 1.):
        feed = self.create_feed_dict(inputs_batch, labels_batch, dropout_prob)
        result = self.sess.run(self.merged, feed_dict=feed)
        self.writer.add_summary(result, index)

    def eveluate_on_batch(self, inputs_batch, labels_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch)
        metric = self.sess.run(self.metric, feed_dict = feed)
        return metric

    def predict_on_batch(self, inputs_batch):
        feed = self.create_feed_dict(inputs_batch)
        predictions = self.sess.run(self.pred, feed_dict=feed)
        return predictions

    def run_evaluate(self, dev_batch):
        scores = 0.0
        iter_num = 0
        for _, x_dev, y_dev in dev_batch:
            score = self.eveluate_on_batch(x_dev, y_dev)
            scores += score
            iter_num += 1
        
        return scores / iter_num

    def build(self):
        self.add_placeholders()
        self.hidden = self.add_conv_op()
        self.logits, self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op()
        self.metric = self.add_F1_op()
        tf.summary.histogram('f1_metric', self.metric)
        self.train_op = self.add_training_op()
        