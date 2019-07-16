#-*- coding:utf8 -*-
import numpy as np
import logging
from keras.utils import to_categorical

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#载入数据函数，如有需要可重载
def load_data(path):
	contents, labels = [], []
	with open(path,'r', encoding='utf-8') as load_f:
		for line in load_f.readlines():
			try:
				content, label = line.strip().split('\t')
				if content:
					contents.append(content)
					labels.append(int(label))
			except:
				pass
	return contents, labels

def get_data_batch(path, batch_size):
	x_data, y_data = load_data(path)
	data_len = len(x_data)
	#logging.info('The number of dataset is {}'.format(data_len))
	y_data = to_categorical(y_data)
	num_batch = int((data_len - 1) / batch_size) + 1
	for i in range(num_batch):
		start_id = i * batch_size
		end_id = min((i + 1) * batch_size, data_len)
		batch_x = x_data[start_id:end_id]
		batch_y = y_data[start_id:end_id]
		yield i, batch_x, batch_y


def process_bert_embeddings(embeddings, sequence_lengths, max_length):
	embeddings_p = []
	for i in range(len(embeddings)):
		sequence_length = sequence_lengths[i]
		# Remove special tokens, i.e. [CLS], [SEP]
		embedding = np.delete(embeddings[i], [0, sequence_length + 1], 0)
		# Remove extra padding tokens
		embedding = np.delete(embedding, np.s_[max_length - 1:-1], 0)
		embeddings_p.append(embedding.tolist())
	return embeddings_p