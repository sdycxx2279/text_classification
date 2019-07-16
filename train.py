import logging
import time
import os

import numpy as np
import tensorflow as tf

from model.textCNN import textCNN

os.environ["CUDA_VISIBLE_DEVICES"] = "1" #设置显卡

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

flags = tf.app.flags
tf.app.flags.DEFINE_string('model', 'cnn','model name')
tf.app.flags.DEFINE_string('train_path', 'data/train.txt','the path of train set')
tf.app.flags.DEFINE_string('dev_path', 'data/dev.txt','the path of validate set')
tf.app.flags.DEFINE_string('output_model', 'output/','the path of model')
tf.app.flags.DEFINE_string('tf_board', 'log/', 'the path of log and tensorboard')
tf.app.flags.DEFINE_integer('max_len', 40, 'max length of input data')
tf.app.flags.DEFINE_integer('epochs', 10, 'epochs')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch_size')
tf.app.flags.DEFINE_integer('n_classes', 10, 'type of labels')
tf.app.flags.DEFINE_integer('checkout_batch_num', 50, 'num of train batch which we evaluate model')
tf.app.flags.DEFINE_float('dropout_prob', 30, 'the parameter of dropout')

flags.DEFINE_string(
    "bert_config_file", "model/models/chinese_L-12_H-768_A-12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")
flags.DEFINE_string(
    "init_checkpoint", "model/models/chinese_L-12_H-768_A-12/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input paragraph. Should be True for uncased "
    "models and False for cased models.")
flags.DEFINE_string("vocab_file", "model/models/chinese_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

FLAGS = tf.app.flags.FLAGS

def main(_):
    logging.info('Start training...')
    with tf.Graph().as_default():
        start = time.time()
        if FLAGS.model == 'cnn':
            logging.info('Initializing CNN...') 
            model = textCNN(
                bert_config_file = FLAGS.bert_config_file,
                init_checkpoint = FLAGS.init_checkpoint,
                vocab_file = FLAGS.vocab_file,
                do_lower_case = FLAGS.do_lower_case,
                max_len = FLAGS.max_len,
                batch_size = FLAGS.batch_size,
                n_epochs = FLAGS.epochs,
                n_classes = FLAGS.n_classes)
        else:
            logging.error('We do not have a model named {}'.format(FLAGS.model))
            return 
    
        logging.info("took %.2f seconds", time.time() - start)
        output_path = FLAGS.output_model + FLAGS.model + '/' + FLAGS.model +'.ckpt'
        #saver = None

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.log_device_placement = True

        with tf.Session(config = sess_config) as sess:
            model.init_sess(sess, FLAGS.tf_board + FLAGS.model + '/train/')
            model.fit(FLAGS.checkout_batch_num, FLAGS.dropout_prob, FLAGS.train_path, FLAGS.dev_path, output_path)
        


if __name__ == '__main__':
    tf.app.run()
