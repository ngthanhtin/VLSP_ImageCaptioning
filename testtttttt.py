import pandas as pd
from json2txt import json2txt
# from googletrans import Translator
# translator = Translator()
# def en2vi(en):
#     translated = translator.translate(en, src='en', dest='vi')
#     print(translated.text)
#     return translated.text

def uppercase(text):
    return text[0].upper() + text[1:]
def add_final_token(text):
    return text + '.'

# json2txt('./submissions/swin_beamsearch2_folds_fold1_0284/results_fold1_0284452.json', './abc.csv')

# df = pd.read_csv('./abc.csv')
# df['captions'] = df['captions'].apply(uppercase)
# df['captions'] = df['captions'].apply(add_final_token)

# import json
# data = []
# for index, row in df.iterrows():
#     captions, id = row['captions'], row['id']
#     data.append({'id': id, 'captions': captions})
# with open('results.json', 'w') as outfile:
#     json.dump(data, outfile, ensure_ascii=False, indent=4)


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
cc = SmoothingFunction()

# true = 'I love you so much'.split()
# print(true)
# pred = 'I love you so much.'.split()

# bleu4 = sentence_bleu([true], pred, smoothing_function=cc.method4)
# print(bleu4)


# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# cc = SmoothingFunction()
# reference = [['this', 'is', 'a', 'test']]
# candidate = ['this', 'is', 'a', 'test', 'test']
# score = sentence_bleu(reference, candidate)
# print(score)


# # very short
# from nltk.translate.bleu_score import sentence_bleu
# reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
# candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy']
# score = sentence_bleu(reference, candidate)
# print(score)

# longer candidate
# from nltk.translate.bleu_score import sentence_bleu
# reference = [['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']]
# candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog', 'from', 'space', 'space', 'space']
# reference = [['a', 'b', 'c']]
# candidate = ['a']
# score = sentence_bleu(reference, candidate, smoothing_function=cc.method4)
# print(score)

