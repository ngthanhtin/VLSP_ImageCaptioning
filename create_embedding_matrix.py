import numpy as np
import torch
from utils import Tokenizer

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
    embed_size = 300
    embedding_matrix = np.zeros((len(tokenizer), embed_size))

    for word, i in tokenizer.stoi.items():

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    with open('embedding_matrix.npy', 'wb') as f:
        np.save(f, embedding_matrix)

    x =  np.load('embedding_matrix.npy')
    # print(x.shape)

tokenizer = torch.load('./tokenizers/tokenizer_vi_fix.pth')
embedding_file = '/home/tinvn/TIN/word2vec_vi_words_300dims.txt'

create_embedding_matrix(tokenizer, embedding_file)