import json
import pandas as pd
file_path = "/home/tinvn/TIN/VLSP_ImageCaptioning/data/vietcap4h-public-test/sample_submission.json"
with open(file_path) as json_file:
    data = json.load(json_file)

all_images = []
all_captions = []

for i in data:
    image_name = i['id']
    
    captions = i['captions']
    captions = captions.split('\n')
    
    for cap in captions:
        all_images.append(image_name)
        all_captions.append(cap)

df = {'id': all_images, 'captions': all_captions}
df = pd.DataFrame(df)

df.to_csv('./test_captions.csv')