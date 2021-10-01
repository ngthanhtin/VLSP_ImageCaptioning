import torch 
import Levenshtein
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import string
import numpy as np
import math
import time

def get_score_levenshtein(y_true, y_pred):
    scores = []
    for true, pred in zip(y_true, y_pred):
        score = Levenshtein.distance(true, pred)
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score

def get_score_bleu(y_true, y_pred):
    cc = SmoothingFunction()
    scores = []
    for true, pred in zip(y_true, y_pred):
        true = true.split()
        pred = pred.split()
        if len(pred) == 1:
            bleu4 = 0.
        else:
            bleu4 = sentence_bleu([true], pred, smoothing_function=cc.method4)
        scores.append(bleu4)
    avg_score = np.mean(scores)
    return avg_score

# Helper functions
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s   = now - since
    es  = s / (percent)
    rs  = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

def lowercase(text_original):
    text_lower = text_original.lower()
    return(text_lower)

def remove_punctuation(text_original):
    text_no_punctuation = text_original.translate(str.maketrans('','',string.punctuation))
    return(text_no_punctuation)

def remove_single_character(text):
    text_len_more_than1 = ""
    for word in text.split():
        if len(word) > 1:
            text_len_more_than1 += " " + word
    return(text_len_more_than1)

def remove_numeric(text,printTF=False):
    text_no_numeric = ""
    for i, word in enumerate(text.split()):
        isalpha = word.isalpha()
        if printTF:
            print("    {:10} : {:}".format(word,isalpha))
        if isalpha:
            if i == 0:
                text_no_numeric += word
            else:
                text_no_numeric += " " + word
        
    return(text_no_numeric)

def fix_error(text_original):
    #-------------------ERROR-----------
    # {"id": "20E8C33538.jpg", "captions": "a man in a hospital gown being carried by a man in a wheelchair
    # co 1 file __NA__: D1E44ACDFF.jpg
    # {"id": "A91D969AB4.jpg", "captions": "Nhóm người đeo khẩu trang đang nói chuyện với nhau qua một chieecs bàn."}
    # {"id": "7966EB6B68.jpg", "captions": "Người đàn ông được một người kiểm tra thân nhiệt.d"}
    # {"id": "FD17DD8D8A.jpg", "captions": "\nNhân viên an ninh áo xanh đang nói chuyện với người phụ nữ áo nâu.viên an ninh."}
    # {"id": "12F3D17CC2.jpg", "captions": "Một nhóm thanh niên đứng ngoài cửa hàng bán đồ lướt sóng Clairemont.
    # {"id": "B990C99058.jpg", "captions": "Có nhiều viên nén trên một bao bì có dòng chữ \"HYDROXYCHLORQUINE 200 MG TAB\"."}
    # {"id": "C82A7C5F48.jpg", "captions": "Một thanh niênđứng ôm vai một người đàn ông lớn tuổi đang ngồi trên ghế."}
    # {"id": "A90B10C5DD.jpg", "captions": "Các nhân viên y tế đang đưa bệnh nhân nằm trêncáng cứu thương lên xe cứu thương."}
    # {"id": "2D31F063C0.jpg", "captions": "Bàn tay cầm một nắm tthuốc đặt cạnh phần bụng một người."}

    # if text_original == "a man in a hospital gown being carried by a man in a wheelchair":
    #     return "Một người đàn ông mặc quần áo bệnh nhân nằm trên xe lăn."

    if text_original == "Nhóm người đeo khẩu trang đang nói chuyện với nhau qua một chieecs bàn.":
        return "Nhóm người đeo khẩu trang đang nói chuyện với nhau qua một chiếc bàn."
    if text_original == "Người đàn ông được một người kiểm tra thân nhiệt.d":
        return "Người đàn ông được một người kiểm tra thân nhiệt."

    if text_original == "Nhân viên an ninh áo xanh đang nói chuyện với người phụ nữ áo nâu.viên an ninh.":
        return "Nhân viên an ninh áo xanh đang nói chuyện với người phụ nữ áo nâu."

    if text_original == "Một nhóm thanh niên đứng ngoài cửa hàng bán đồ lướt sóng Clairemont.":
        return "Một nhóm thanh niên đứng ngoài cửa hàng bán đồ lướt sóng Clairemont."
    if text_original == "Có nhiều viên nén trên một bao bì có dòng chữ \"HYDROXYCHLORQUINE 200 MG TAB\".":
        return "Có nhiều viên nén trên một bao bì có dòng chữ \"HYDROXYCHLORQUINE 200 MG TAB\"."

    if text_original == "Một thanh niênđứng ôm vai một người đàn ông lớn tuổi đang ngồi trên ghế.":
        return "Một thanh niên đứng ôm vai một người đàn ông lớn tuổi đang ngồi trên ghế."
    if text_original == "Các nhân viên y tế đang đưa bệnh nhân nằm trêncáng cứu thương lên xe cứu thương.":
        return "Các nhân viên y tế đang đưa bệnh nhân nằm trên cáng cứu thương lên xe cứu thương."
    if text_original == "Bàn tay cầm một nắm tthuốc đặt cạnh phần bụng một người.":
        return "Bàn tay cầm một nắm thuốc đặt cạnh phần bụng một người."

    else:
        return text_original

def text_clean(text_original):
    text = lowercase(text_original)
    if 'covid-19' in text or 'n95' in text or '200' in text: # fix special nouns
        return (text[:-1]) # remove '.'
    text = remove_punctuation(text)
    # text = remove_single_character(text)
    text = remove_numeric(text)
    return(text)

class Tokenizer(object):
    
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.stofreq = {}

    def __len__(self):
        return len(self.stoi)
    
    def fit_on_texts(self, texts):
        vocab = set()
        for text in texts:
            vocab.update(text.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            # count frequency
            if s not in self.stofreq:
                self.stofreq[s] = 1
            else:
                self.stofreq[s] += 1
                
            self.stoi[s] = i
        self.itos = {item[1]: item[0] for item in self.stoi.items()}
        
    def text_to_sequence(self, text):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        for s in text.split(' '):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence
    
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
        return sequences

    def sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))
    
    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = self.sequence_to_text(sequence)
            texts.append(text)
        return texts
    
    def predict_caption(self, sequence):
        caption = ''
        for i in sequence:
            if i == self.stoi['<eos>'] or i == self.stoi['<pad>']:
                break
            caption += ' ' + self.itos[i]
        return caption
    
    def predict_captions(self, sequences):
        captions = []
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
        return captions



