import logging
import time
import os

import numpy as np
import tensorflow as tf

from model.textCNN import textCNN
from utils import get_data_batch

os.environ["CUDA_VISIBLE_DEVICES"] = "1" #设置显卡

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

flags = tf.app.flags
tf.app.flags.DEFINE_string('model', 'cnn','model name')
tf.app.flags.DEFINE_string('train_path', 'data/train.txt','the path of train set')
tf.app.flags.DEFINE_string('val_path', 'data/val.txt','the path of validate set')
tf.app.flags.DEFINE_string('output_model', 'output/','the path of model')
tf.app.flags.DEFINE_string('tf_board', 'log/', 'the path of log and tensorboard')
tf.app.flags.DEFINE_integer('max_len', 40, 'max length of input data')
tf.app.flags.DEFINE_integer('epochs', 10, 'epochs')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch_size')
tf.app.flags.DEFINE_integer('n_classes', 10, 'type of labels')
tf.app.flags.DEFINE_integer('checkout_batch_num', 30, 'num of train batch which we evaluate model')
tf.app.flags.DEFINE_float('dropout_prob', 30, 'the parameter of dropout')

FLAGS = tf.app.flags.FLAGS

def main(_):
    logging.info('Loading data...')
    data_train = get_data_batch('data/train.txt', FLAGS.batch_size)
    data_val = get_data_batch('data/val.txt', FLAGS.batch_size)
    logging.info('Start training...')
    with tf.Graph().as_default():
        start = time.time()
        if FLAGS.model == 'cnn':
            logging.info('Initializing CNN...') 
            model = textCNN(
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
            model.fit(FLAGS.checkout_batch_num, FLAGS.dropout_prob, data_train, data_val, output_path)
        


if __name__ == '__main__':
    tf.app.run()
