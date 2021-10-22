import numpy as np
import glob
import os
import pandas as pd
import datefinder
import cv2

def correct(text,field="total"):

    correct_patterns = {
        'TIÊN': 'TIỀN',
        'TTỔNG': 'TỔNG',
        'Tiên': 'Tiền',
        'TIẾN': 'TIỀN',
        '11ỀN': 'TIỀN',
        'TSTOÁN': 'T. TOÁN',
        'tiên': 'tiền',
        'toc': 'toán',
        'QUẤY': 'QUẦY',
        'CŨNG': 'CỘNG',
        'Tiến': 'Tiền',
        'OÁN': 'TOÁN',
        'TĂNG': 'TỔNG',
        'TOÁNG': 'TOÁN',
        'TỜNG': 'TỔNG',
        'TÔNG': 'TỔNG',
        'toàn': 'toán',
        'QUÁY': 'QUẦY',
        'T,': 'T.',
        'TOÀN': 'TOÁN',
        'Tpán': 'Toán',
        'CÔNG': 'CỘNG',
        'Ống': 'Tổng',
        'Cộng': 'Cộng',
        'tiến': 'tiền',
        'toan(VND)': 'toan (VND)',
        'stong': 'tong',
        'CŨNG:': 'CỘNG:',
        "TŨNG" :"TỔNG",
        # '-': '',
        '1': '',
        'VÀ': '',
        'tiên:': 'tiền:',
        'QUẤY':'QUẦY',
        '"TỔNG': 'TỔNG',
        'TOÁN"': 'TOÁN'
    }

    date_patterns = {
        'bản': 'bán',
        'Ngayr': 'Ngay:',
        'họn': 'hẹn',
        'bản:': 'bán:',
        'Bản:': 'Bán:',
        'iểm': 'Điểm',

    }

    new_line = text.strip()
    if field =="total":
        for word in text.strip().split(' '):
            if word in correct_patterns: # date_patterns for date 
                new_line = new_line.replace(word, correct_patterns.get(word)) # date_patterns for date 
    if field =="date":
        for word in text.strip().split(' '):
            if word in date_patterns: # date_patterns for date 
                new_line = new_line.replace(word, date_patterns.get(word)) # date_patterns for date 
        
    return new_line

paths = glob.glob("results/*")
node_labels = ['other', 'company', 'address', 'date', 'total']
def hasNumbers(inputString):
    inputString = inputString.replace("Đ","")
    return any(char.isdigit() for char in inputString)

def hasCharacters(inputString):
    return any(char.isalpha() for char in inputString)
color = (0, 0, 255)

data_frame = []
count = 0
for path in paths :
    results = {"company":[],"address":[],"date":[],"total":[]}
    outputs   = {"company":"","address":"","date":"","total":""}
    name = os.path.basename(path)
    name_id = name.replace("txt","jpg")
    with open(path,"r") as file :
        data = file.readlines()
        for line in data:
            tmp = line.strip().split("\t")
            x,y = tmp[:2]
            ymax = tmp[5]
            text = tmp[10]
            class_id = tmp[11]
            save = [int(x),int(y),text,class_id,ymax]
            results[class_id].append(save)      
        for key,v in results.items():       
            if key == "date" :
                for j in results[key]:
                    if "Ngày" in j[2] :
                        j[2] =  "Ngày" + j[2].split("Ngày")[1]
                    if "Thời" in j[2] :
                        j[2] =  "Thời" + j[2].split("Thời")[1]
                if len(results[key])==0:
                    with open(f"ocr_parser/{name}","r") as f :
                        file = f.readlines()
                        for i in file :
                            x1,y1 = i.strip().split("\t")[:2]
                            tm = i.strip().split("\t")[8]
                            tm2 = tm.split(" ")
                            for k in tm2 :
                                if len(k)<10 and (k.count("-") <2 or k.count("/")<2  or k.count(".")<2):
                                    continue
                                else :
                                    try  :
                                        matches = datefinder.find_dates(k)
                                        for match in matches:
#                                             print(tm)           
                                            if "Ngày" in tm:
                                                
                                                results[key].append([int(x1),int(y1),"Ngày" + tm.split("Ngày")[1],"date",ymax])
            
#                                                 break
                                            elif "Thời" in tm :
                                                results[key].append([int(x1),int(y1),"Thời" + tm.split("Thời")[1],"date",ymax])
#                                                 break
                                            else :
                                                results[key].append([int(x1),int(y1),tm,"date",ymax])
#                                                 break
                                    except :
                                        continue
            
            
            
        if len(results["total"]) == 1 :  
            if (hasNumbers(results["total"][0][2])==False or hasCharacters(results["total"][0][2])==False) :
                box1 = results["total"][0]
                ymin_1,ymax_1 = int(box1[1]),int(box1[-1])
                yc_c1 = (ymax_1 + ymin_1)/2
                with open(f"ocr_parser/{name}","r") as f :
                    file = f.readlines()
                    same_line = []
                    for i in file :
                        bboxs = i.strip().split("\t")
                        label_line = bboxs[8]
                        ymin_2,ymax_2 = int(bboxs[1]),int(bboxs[5])
                        yc_c2 = (ymax_2 + ymin_2)/2
                        if abs(yc_c1-yc_c2) < 10 and (label_line.lower() not in results["total"][0][2].lower()):
#                             results["total"].append([int(bboxs[0]),ymin_2,label_line,ymax_2])
                            same_line.append([int(bboxs[0]),ymin_2,label_line,ymax_2])
                    if len(same_line)==1:
                            results["total"].append(same_line[0])                           
    #sorted(my_list , key=lambda k: [k[1], k[0]])
    company = "|||".join([str(i[2]) for i in sorted(results["company"] , key=lambda k: k[1])]) 
    adress = "|||".join([correct(str(i[2]),field="address") for i in sorted(results["address"] , key=lambda k: k[1])])
#     print(results["date"])
    date = "|||".join([correct(str(i[2]),field="date") for i in sorted(results["date"] , key=lambda k: k[0])])
#     print("--->",date)
    total = "|||".join([correct(str(i[2]),field="total") for i in sorted(results["total"] , key=lambda k: k[0])])
    labels_save =  company+"|||"+ adress + "|||" + date + "|||" + total
    # print("="*20,name_id,"="*20)
    # print(labels_save)
    # print("-"*50)
    data_frame.append([name_id,0.5,labels_save])
# data_frame.append(["mcocr_val_145114unyae.jpg",0.5,"|||||||||"])
columns = ["img_id","anno_image_quality","anno_texts"]
df = pd.DataFrame(data_frame,columns=columns)
df1 = pd.read_csv("mcocr_test_samples_df.csv")
df.to_csv("results.csv",index=None,columns=columns)

df_final = df1.reset_index()[['index', 'img_id']].merge(df, on='img_id', how='left').fillna('|||||||||').sort_values(by='index').drop(columns='index')
df_final.to_csv("submit/results.csv",index=None)
# df1 = pd.read_csv("mcocr_test_samples_df.csv")
# df2 = pd.read_csv("results.csv")
# df_final.to_csv("submit_final/results.csv",index=None)
