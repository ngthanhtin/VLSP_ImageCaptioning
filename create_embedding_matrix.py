import numpy as np
import torch
from utils import Tokenizer

embed_size = 300

def create_embedding_matrix(tokenizer, embedding_file):
    """
    Load pretrained embedding and output the npy contains the pretrained vectors
    """
    embeddings_index = {}
    with open(embedding_file, encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # specify the embedding size
    embedding_matrix = np.zeros((len(tokenizer), embed_size))

    for word, i in tokenizer.stoi.items():

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    with open('embedding_matrix2.npy', 'wb') as f:
        np.save(f, embedding_matrix)

    

tokenizer = torch.load('./tokenizers/tokenizer_vi_fix.pth')
# embedding_file = '/home/tinvn/TIN/word2vec_vi_words_300dims.txt'
embedding_file = '/home/tinvn/TIN/VLSP_ImageCaptioning/vlsp_code/cc.vi.300.vec'

# create_embedding_matrix(tokenizer, embedding_file)

x =  np.load('embedding_matrix2.npy')
zeros = np.zeros((embed_size))

num_zeros = 0
for word, i in tokenizer.stoi.items():
    if (zeros[0] == x[i]).all():
        num_zeros += 1
        continue

print(num_zeros)