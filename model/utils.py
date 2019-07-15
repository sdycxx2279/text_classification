import numpy as np

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