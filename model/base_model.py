import os
import tensorflow as tf

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class BaseModel(object):
	"""Generic class for general methods"""

	def __init__(self, config):
		"""Defines self.config
		Args:
			config: (Config instance) class with hyper parameters,
				vocab and embeddings
		"""
		self.config = config
		self.sess = None
		self.saver = None

	def reinitialize_weights(self, scope_name):
		"""Reinitializes the weights of a given layer"""
		variables = tf.contrib.framework.get_variables(scope_name)
		init = tf.variables_initializer(variables)
		self.sess.run(init)

	def train_on_batch(self, inputs_batch, labels_batch, dropout_prob = 1.):
		raise NotImplementedError

	def board_on_batch(self, inputs_batch, labels_batch, index, dropout_prob = 1.):
		raise NotImplementedError

	def eveluate_on_batch(self, inputs_batch, labels_batch):
		raise NotImplementedError

	def predict_on_batch(self, inputs_batch):
		raise NotImplementedError

	def run_evaluate(self, dev_batch):
		raise NotImplementedError


	def init_sess(self, sess, tb_path):
		self.sess = sess
		self.merged = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(tb_path, self.sess.graph)

		init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		self.saver = tf.train.Saver()
		self.sess.run(init)

	def fit(self, check_num, dropout_prob, data_batch, dev_batch, output_model):
		best_score = 0.
		not_check_num = 0
		for epoch in range(self.config.n_epochs):
			logging.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
			for index, batch_x, batch_y in data_batch:
				loss = self.train_on_batch(batch_x, batch_y, dropout_prob)
				#logging.info("Loss:{}".format(loss))
				if index % check_num == 0:
					logging.info('The loss is {}'.format(loss))
					self.board_on_batch(batch_x, batch_y, index, dropout_prob)
					not_check_num += 1
					score = self.run_evaluate(dev_batch)
					logging.info('The score of evaluate is {}'.format(score))
					if score > best_score:
						best_score = score
						not_check_num = 0
						if self.saver:
							logging.info("New best score! Saving model in %s", output_model)
							self.saver.save(self.sess, output_model)

						if not_check_num == check_num:
							logging.info("The model has not improve at {} batches, training is over.".format(check_num))
							break

				logging.info("The best score of model is {} in Epoch {}".format(best_score, epoch + 1))