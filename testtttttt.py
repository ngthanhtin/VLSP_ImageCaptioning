import pandas as pd
import json

df = pd.read_csv('submission.csv')
data = []
for index, row in df.iterrows():
    captions, id = row['captions'], row['id']
    data.append({'id': id, 'captions': captions})
with open('submission.json', 'w') as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=4)