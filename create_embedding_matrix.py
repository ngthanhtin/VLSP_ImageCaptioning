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

    

tokenizer = torch.load('./tokenizers/tokenizer_vi_fix_spelling.pth')
zero_indexes = [0, 467, 490, 494, 563, 564, 570, 973, 1176, 1281, 1455, 1609, 1610, 1611]
for i in zero_indexes:
    print(tokenizer.itos[i])

# embedding_file = './pretrained_embedding/word2vec_vi_words_300dims.txt'
embedding_file = './pretrained_embedding/cc.vi.300.vec'

# create_embedding_matrix(tokenizer, embedding_file)

x =  np.load('./pretrained_embedding/embedding_matrix2.npy')
zeros = np.zeros((embed_size))

num_zeros = 0
zero_indexes = []
for word, i in tokenizer.stoi.items():
    if (zeros[0] == x[i]).all():
        num_zeros += 1
        zero_indexes.append(i)
        continue

print(num_zeros)
print(zero_indexes)