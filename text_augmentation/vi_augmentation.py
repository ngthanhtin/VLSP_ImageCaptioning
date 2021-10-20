

from nltk import word_tokenize
from pyvi import ViTokenizer, ViPosTagger
import random
import csv
import pandas as pd
from utils import fix_error, text_clean
from tqdm.auto import tqdm

#====================================================================================================
# Hàm hoán đổi ngẫu nhiên vị trí các từ trong câu. Số lượng câu mới tạo ra là n

def Random_Swap(sentence,n):
    new_sentences = []
    words = sentence.split()
    for i in range(n):
        random.shuffle(words)
        new_sentences.append(' '.join(words))
    return new_sentences

#====================================================================================================
# Hàm xoá từ trong câu nếu từ đó là: Động từ (V), giới từ (C), số (M), dấu câu (F)

def Random_Deletion(sentence):
    
    new_sentence = []
    tagged_word = ViPosTagger.postagging(sentence)
    
    # Chia POS thành 2 list cho dễ thực hiện
    word = tagged_word[0]
    tag = tagged_word[1]

    edited_sentence = [i for i,j in zip(word,tag) if j != 'P' and j != 'V' and j != 'C' and j != 'F' and j != 'M']
    edited_sentence = ' '.join(edited_sentence)
    new_sentence.append(edited_sentence)
    return new_sentence



# Đoạn code này là đọc tất cả các comment và label. Thực hiện agumentation.
# Sau đó lưu vào file .csv
  
df = pd.read_csv('./train_files/train_captions.csv')

header=['id','captions']
with open('./train_files/train_captions_2.csv', 'a', encoding='utf8') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for i in range(len(df) - 1):
        id = df['id'][i]
        captions = df['captions'][i]
        captions = fix_error(captions)
        last_token = captions[-1]
        if last_token == '.':
            captions = captions[:-1]

            data = [id,captions + '.']
            writer.writerow(data)
        
            # Thực hiện swapping
            data= [id, Random_Swap(captions,1)[0] + '.']
            writer.writerow(data)
            
            # Thực hiện việc xoá ngẫu nhiên
            data= [id, Random_Deletion(captions)[0] + '.']
            writer.writerow(data)
        else:
            
            data = [id,captions]
            writer.writerow(data)
        
            # Thực hiện swapping
            data= [id, Random_Swap(captions,1)[0]]
            writer.writerow(data)
            
            # Thực hiện việc xoá ngẫu nhiên
            data= [id, Random_Deletion(captions)[0]]
            writer.writerow(data)

        
        