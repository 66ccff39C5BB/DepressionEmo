import argparse
import time
import pandas as pd
import numpy as np

import lightgbm as lgbm
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from file_io import *

emotion_list = ['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness', 'loneliness', 'sadness', 'suicide intent', 'worthlessness']
symptom_list = ['Sadness', 'Pessimism', 'Sense_of_failure', 'Loss_of_Pleasure', 'Guilty_feelings', 'Sense_of_punishment', 'Self-dislike', 'Self-incrimination', 'Suicidal_ideas', 'Crying', 'Agitation', 'Social_withdrawal', 'Indecision', 'Feelings_of_worthlessness', 'Loss_of_energy', 'Change_of_sleep', 'Irritability', 'Changes_in_appetite', 'Concentration_difficulty', 'Tiredness_or_fatigue', 'Loss_of_interest_in_sex']


current_time = time.strftime("%Y%m%d%H%M%S")
parser = argparse.ArgumentParser(description='Training Parameter')
parser.add_argument('--train_path', type=str, default='Dataset/train.json') 
parser.add_argument('--test_path', type=str, default='Dataset/test.json')
parser.add_argument('--val_path', type=str, default='Dataset/val.json')
args = parser.parse_args()

label_list = symptom_list if "BDISen" in args.test_path else emotion_list

    
    
train_set = read_list_from_jsonl_file(args.train_path)
val_set = read_list_from_jsonl_file(args.val_path)
test_set = read_list_from_jsonl_file(args.test_path)

dataset = train_set + val_set + test_set

data_labels = [[] for i in range(len(label_list))]
for item in dataset:
    data_label = str(item['label_id'])
    while len(data_label) < len(label_list):
        data_label = '0' + data_label
    for i in range(len(data_label)):
        data_labels[i].append(float(data_label[i]))

df_labels = np.concatenate([np.array(data_labels[i])[np.newaxis, :] for i in range(len(label_list))], axis=0)

data = []
for item in dataset:
    text = item['text']        
    data.append(text)

vectorizer = TfidfVectorizer()
BOW = vectorizer.fit_transform(data)

x_train, y_train = BOW[:len(train_set)], df_labels[:, :len(train_set)]
x_val, y_val = BOW[len(train_set):len(train_set)+len(val_set)], df_labels[:, len(train_set):len(train_set)+len(val_set)]
x_test, y_test = BOW[len(train_set)+len(val_set):], df_labels[:, len(train_set)+len(val_set):]

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

models = [LGBMClassifier(objective='binary', learning_rate = 0.5, verbose = -1) for i in range(len(label_list))]

for i in range(len(label_list)):
    models[i].fit(x_train,y_train[i])

predictions = []
for i in range(len(label_list)):
    predictions.append(np.array(models[i].predict(x_test))[np.newaxis, :])
predictions = np.concatenate(predictions, axis=0)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

f1_mi = f1_score(y_true=y_test, y_pred=predictions, average='micro')
re_mi = recall_score(y_true=y_test, y_pred=predictions, average='micro')
pre_mi = precision_score(y_true=y_test, y_pred=predictions, average='micro')
    
f1_mac = f1_score(y_true=y_test, y_pred=predictions, average='macro')
re_mac = recall_score(y_true=y_test, y_pred=predictions, average='macro')
pre_mac = precision_score(y_true=y_test, y_pred=predictions, average='macro')
    
result = {}
result['f1_micro'] = f1_mi
result['recall_micro'] = re_mi
result['precision_micro'] = pre_mi
    
result['f1_macro'] = f1_mac
result['recall_macro'] = re_mac
result['precision_macro'] = pre_mac

print(result)