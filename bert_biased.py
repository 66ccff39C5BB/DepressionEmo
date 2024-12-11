import argparse
import gc
import os
import jieba
import jieba.analyse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import sys
import torch
import transformers
import time
import wandb

from collections import defaultdict
from file_io import *
from matplotlib import rc
from pylab import rcParams
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from textwrap import wrap
from torch import nn, optim, functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
emotion_list = ['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness', 'loneliness', 'sadness', 'suicide intent', 'worthlessness']
symptom_list = ['Sadness', 'Pessimism', 'Sense_of_failure', 'Loss_of_Pleasure', 'Guilty_feelings', 'Sense_of_punishment', 'Self-dislike', 'Self-incrimination', 'Suicidal_ideas', 'Crying', 'Agitation', 'Social_withdrawal', 'Indecision', 'Feelings_of_worthlessness', 'Loss_of_energy', 'Change_of_sleep', 'Irritability', 'Changes_in_appetite', 'Concentration_difficulty', 'Tiredness_or_fatigue', 'Loss_of_interest_in_sex']

class PreparedDataset():
    def __init__(self, texts, categories, tokenizer, max_len):
        self.texts = texts
        self.categories = categories
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        category = self.categories[item]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            #pad_to_max_length=True,
            padding = "max_length",
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
            )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'categories': torch.tensor(category, dtype=torch.float)
        }

class CategoryClassifier(nn.Module):
    def __init__(self, n_labels, pretrained_model = 'bert-base-cased'):
        super(CategoryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, n_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, return_dict=False):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        output = self.drop(pooled_output)
        output = self.out(output)
        output = self.sigmoid(output)
        return output

def search_index(type_list, class_names):

    for index, c in enumerate(class_names):
        c_list = c
        
        if (type(c) is tuple): c_list = [*c]
        else: c_list = [c_list]
        
        #print('-- c_list: ', c_list)
        n_intersection = len(set(c_list).intersection(set(type_list)))
        #print('-- n_intersection: ', n_intersection)
        if (n_intersection == len(c_list) and len(c_list) == len(type_list)): return index

    return -1

def create_data_loader(dataset, tokenizer, label_names, max_len, batch_size):

    texts, categories = [], []
 
    for item in dataset:
        item_classifier = str(item['label_id'])
        while len(item_classifier) < len(label_names): item_classifier = '0' + item_classifier
        label = [int(x) for x in item_classifier]
        assert len(label) == len(label_names)
        categories.append(label)
        texts.append(item['text'])
    
    #print('categories: ', categories)
    ds = PreparedDataset(texts=np.array(texts),
                         categories=np.array(categories),
                         tokenizer=tokenizer,
                         max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=4)


def train_epoch(biased_model, debiased_model, data_loader, val_data_loader, biased_loss_fn, loss_fn, biased_optimizer, debiased_optimizer, device, biased_scheduler, debiased_scheduler, n_examples, cur_threshold):

    biased_model = biased_model.train()
    debiased_model = debiased_model.train()
    losses = []
    i = 0
    pred_list = []
    true_list = []
    cur_threshold = torch.tensor(cur_threshold).to(device)
    
    for d in data_loader:
        
        # sys.stdout.write('Training batch: %d/%d \r' % (i, len(data_loader)))
        #sys.stdout.flush()
        
        i = i + 1
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        categories = d["categories"].to(device)
        biased_outputs = biased_model(input_ids=input_ids, attention_mask=attention_mask)
        outputs = debiased_model(input_ids=input_ids, attention_mask=attention_mask)
        preds = (outputs > cur_threshold).to(torch.float)

        
        biased_loss = biased_loss_fn(biased_outputs, categories)

        biased_loss.backward()
        nn.utils.clip_grad_norm_(biased_model.parameters(), max_norm=1.0)
        biased_optimizer.step()
        biased_scheduler.step()
        biased_optimizer.zero_grad()


        loss = loss_fn(biased_outputs, outputs, categories)

        loss.backward()
        nn.utils.clip_grad_norm_(debiased_model.parameters(), max_norm=1.0)
        debiased_optimizer.step()
        debiased_scheduler.step()
        debiased_optimizer.zero_grad()

        
        losses.append(loss.item())
        pred_list.append(preds.tolist())
        true_list.append(categories.tolist())

    pred_list = sum(pred_list, [])
    true_list = sum(true_list, [])
    
    f1_mi = f1_score(y_true=true_list, y_pred=pred_list, average='micro')
    re_mi = recall_score(y_true=true_list, y_pred=pred_list, average='micro')
    pre_mi = precision_score(y_true=true_list, y_pred=pred_list, average='micro')
    
    f1_mac = f1_score(y_true=true_list, y_pred=pred_list, average='macro')
    re_mac = recall_score(y_true=true_list, y_pred=pred_list, average='macro')
    pre_mac = precision_score(y_true=true_list, y_pred=pred_list, average='macro')
    
    result = {}
    result['f1_micro'] = f1_mi
    result['recall_micro'] = re_mi
    result['precision_micro'] = pre_mi
    
    result['f1_macro'] = f1_mac
    result['recall_macro'] = re_mac
    result['precision_macro'] = pre_mac

    return result, np.mean(losses)
        
    #return correct_predictions.double() / n_examples, np.mean(losses)

 
def eval_model(model, data_loader, loss_fn, device):
    model = model.eval()

    outputs = []
    true_classes = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            categories = d["categories"].to(device)
            
            outputs.append(model(input_ids=input_ids, attention_mask=attention_mask))
            true_classes.append(categories)


    best_threshold = [0.5 for _ in label_list]
    best_threshold = torch.tensor(best_threshold).to(device)

    best_dev_cmaf1, best_dev_cmarec, best_dev_cmapre = -99999999, -99999999, -99999999
    best_dev_cmif1, best_dev_cmirec, best_dev_cmipre = -99999999, -99999999, -99999999
    best_mean_loss = 99999999

    # not search the best threshold in this version
    cur_threshold = best_threshold.clone().detach()

    losses = []    
    pred_list = []
    true_list = []
    
    for o, categories in zip(outputs, true_classes):
        preds = (o > cur_threshold).to(torch.float)

        loss = loss_fn(o, categories)
        pred_list.append(preds.tolist())
        true_list.append(categories.tolist())
        losses.append(loss.item())

    pred_list = sum(pred_list, [])
    true_list = sum(true_list, [])
    for main_index in range(len(label_list)):
        main_pred_list = [pred[main_index] for pred in pred_list]
        main_true_list = [true[main_index] for true in true_list]
        main_f1 = f1_score(y_true=main_true_list, y_pred=main_pred_list)
        main_re = recall_score(y_true=main_true_list, y_pred=main_pred_list)
        main_pre = precision_score(y_true=main_true_list, y_pred=main_pred_list)
        wandb.log({label_list[main_index] + '_f1/target_f1': main_f1, label_list[main_index] + '_recall/target_recall': main_re, label_list[main_index] + '_precision/target_precision': main_pre})
        for other_index in range(len(label_list)):
            if main_index == other_index: continue
            other_true_list = [true[other_index] for true in true_list]
            other_f1 = f1_score(y_true=other_true_list, y_pred=main_pred_list)
            other_re = recall_score(y_true=other_true_list, y_pred=main_pred_list)
            other_pre = precision_score(y_true=other_true_list, y_pred=main_pred_list)
            wandb.log({label_list[main_index] + '_f1/' + label_list[other_index] + '_f1': other_f1, label_list[main_index] + '_recall/' + label_list[other_index] + '_recall': other_re, label_list[main_index] + '_precision/' + label_list[other_index] + '_precision': other_pre})
    
    f1_mi = f1_score(y_true=true_list, y_pred=pred_list, average='micro')
    re_mi = recall_score(y_true=true_list, y_pred=pred_list, average='micro')
    pre_mi = precision_score(y_true=true_list, y_pred=pred_list, average='micro')
    
    f1_mac = f1_score(y_true=true_list, y_pred=pred_list, average='macro')
    re_mac = recall_score(y_true=true_list, y_pred=pred_list, average='macro')
    pre_mac = precision_score(y_true=true_list, y_pred=pred_list, average='macro')
    
    if f1_mac + f1_mi > best_dev_cmaf1 + best_dev_cmif1:
        best_threshold = cur_threshold.clone().detach()
        best_dev_cmaf1, best_dev_cmarec, best_dev_cmapre = f1_mac, re_mac, pre_mac
        best_dev_cmif1, best_dev_cmirec, best_dev_cmipre = f1_mi, re_mi, pre_mi
        best_mean_loss = np.mean(losses)
        
        result = {}
        result['f1_micro'] = best_dev_cmif1
        result['recall_micro'] = best_dev_cmirec
        result['precision_micro'] = best_dev_cmipre

        result['f1_macro'] = best_dev_cmaf1
        result['recall_macro'] = best_dev_cmarec
        result['precision_macro'] = best_dev_cmapre

        result['best_threshold'] = best_threshold.tolist()

        return result, best_mean_loss

def get_predictions(model, data_loader, best_threshold):
    model = model.eval()
    sentences = []
    predictions = []
    prediction_probs = []
    real_values = []
    best_threshold = torch.tensor(best_threshold).to(device)

    with torch.no_grad():
        for d in data_loader:
            texts = d["text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            categories = d["categories"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = (outputs > best_threshold).to(torch.float)
            sentences.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(categories)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return sentences, predictions, prediction_probs, real_values


def train_model(train_set, val_set, pretrained_model = 'bert-base-cased', 
                saved_model_file = 'best_model_state.bin', saved_history_file = 'history_file.json', saved_best_threshold = 'best_threshold.json',
                epochs = 40, max_len = 256, batch_size = 8, q_loss = 0.5):

    # prepare dataset
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    train_data_loader = create_data_loader(train_set, tokenizer, label_list, max_len, batch_size)
    val_data_loader = create_data_loader(val_set, tokenizer, label_list, max_len, batch_size)
    #test_data_loader = create_data_loader(df_test, tokenizer, class_names, max_len, batch_size)

    # create model
    biased_model = CategoryClassifier(len(label_list), pretrained_model)
    biased_model = biased_model.to(device)
    debiased_model = CategoryClassifier(len(label_list), pretrained_model)
    debiased_model = debiased_model.to(device)

    # data = next(iter(train_data_loader))
    # input_ids = data['input_ids'].to(device)
    # attention_mask = data['attention_mask'].to(device)
    #F.softmax(model(input_ids, attention_mask), dim=1)
    
    # lr=2e-5
    biased_optimizer = AdamW(biased_model.parameters(), lr=2e-5, correct_bias=True)
    total_steps = len(train_data_loader) * epochs
    biased_scheduler = get_linear_schedule_with_warmup(
        biased_optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
        )

    debiased_optimizer = AdamW(debiased_model.parameters(), lr=2e-5, correct_bias=True)
    debiased_scheduler = get_linear_schedule_with_warmup(
        debiased_optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
        )
    
    class GeneralizedCrossEntropy(nn.Module):
        def __init__(self, q=0.5):
            super(GeneralizedCrossEntropy, self).__init__()
            self.q = q

        def forward(self, inputs, targets):
            loss = (1 - torch.pow(inputs, self.q)) * targets / self.q
            loss = loss.sum(dim=1).mean()
            return loss
    
    class WeightedCrossEntropy(nn.Module):
        def __init__(self):
            super(WeightedCrossEntropy, self).__init__()

        def cross_entropy(self, inputs, targets):
            # without sum
            return -targets * torch.log(inputs)

        def forward(self, bias_inputs, inputs, targets):
            bias_loss = self.cross_entropy(bias_inputs, targets)
            debias_loss = self.cross_entropy(inputs, targets)
            weight = bias_loss / (bias_loss + debias_loss)
            loss = weight.detach() * debias_loss
            return loss.sum(dim=1).mean()
        
    biased_loss_fn = GeneralizedCrossEntropy(q_loss).to(device)
    loss_fn = WeightedCrossEntropy().to(device)

    history = defaultdict(list)
    best_metric = 0
    best_epoch = -1
    write_single_dict_to_json_file(saved_history_file, {}, "w")
    cur_threshold = [0.5 for _ in label_list]
    
    for epoch in range(epochs):
        print('-' * 50)
        print(f'Epoch {epoch + 1}/{epochs}')
        

        train_result, train_loss = train_epoch(biased_model, debiased_model, train_data_loader, val_data_loader, biased_loss_fn, loss_fn, biased_optimizer, debiased_optimizer, device, biased_scheduler, debiased_scheduler, len(train_set), cur_threshold = cur_threshold)
        train_f1_macro = train_result['f1_macro']
        print(f'Train loss: {train_loss}, Train f1 macro: {train_f1_macro}')
        
        val_result, val_loss = eval_model(debiased_model, val_data_loader, nn.CrossEntropyLoss().to(device), device)

        cur_threshold = val_result['best_threshold']
        
        f1_macro = val_result['f1_macro']
        f1_micro = val_result['f1_micro']
        print(f'Val loss {val_loss}, Val f1 macro: {f1_macro}, Val f1 micro: {f1_micro}')

        wandb.log(val_result)
        
        history['train_result'].append(train_result)
        history['train_loss'].append(train_loss)
        history['val_result'].append(val_result)
        history['val_loss'].append(val_loss)
      
        if f1_macro + f1_micro > best_metric:
            biased_save_path = 'bert-base-biased/' + current_time + '/best_bert_model.bin'
            if not os.path.exists('bert-base-biased/' + current_time): os.makedirs('bert-base-biased/' + current_time)
            torch.save(biased_model.state_dict(), biased_save_path)
            torch.save(debiased_model.state_dict(), saved_model_file)
            json.dump(cur_threshold, open(saved_best_threshold, 'w+'))
            print('Model saved.')
            best_metric = f1_macro + f1_micro
            best_epoch = epoch + 1

        # save history step
        print('val_f1_macro: ', f1_macro)
        print('train_f1_mac: ', train_f1_macro)

        history_dict = {}
        history_dict['epoch'] = str(epoch + 1)
        history_dict['best_epoch'] = str(best_epoch)
        history_dict['best_metric'] = str(best_metric)
        history_dict['train_result'] = train_result
        history_dict['train_loss'] = train_loss
        history_dict['val_result'] = val_result
        history_dict['val_loss'] = val_loss
        write_single_dict_to_json_file(saved_history_file, history_dict)
        
        torch.cuda.empty_cache()
        gc.collect()
        print('-' * 50)
    
 
def test_dataset(test_set, 
                pretrained_model = 'bert-base-cased', saved_model_file = 'best_bert_model.bin', saved_best_threshold = 'best_threshold.json',
                saved_history_file = 'best_bert_model.json', max_len = 256, batch_size = 8):
    
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    best_threshold = json.load(open(saved_best_threshold, 'r'))

    #train_data_loader = create_data_loader(train_set, tokenizer, class_names, max_len, batch_size)
    #val_data_loader = create_data_loader(val_set, tokenizer, class_names, max_len, batch_size)
    test_data_loader = create_data_loader(test_set, tokenizer, label_list, max_len, batch_size)

    loss_fn = nn.CrossEntropyLoss().to(device)

    model = CategoryClassifier(len(label_list), pretrained_model)
    model.load_state_dict(torch.load(saved_model_file))
    model = model.to(device)
    
    texts, pred_list, pred_probs, true_list = get_predictions(model, test_data_loader, best_threshold)
   
    pred_list = pred_list.tolist()
    true_list = true_list.tolist()
    
    f1_mi = f1_score(y_true=true_list, y_pred=pred_list, average='micro')
    re_mi = recall_score(y_true=true_list, y_pred=pred_list, average='micro')
    pre_mi = precision_score(y_true=true_list, y_pred=pred_list, average='micro')
    
    f1_mac = f1_score(y_true=true_list, y_pred=pred_list, average='macro')
    re_mac = recall_score(y_true=true_list, y_pred=pred_list, average='macro')
    pre_mac = precision_score(y_true=true_list, y_pred=pred_list, average='macro')
    
    result = {}
    result['f1_micro'] = f1_mi
    result['recall_micro'] = re_mi
    result['precision_micro'] = pre_mi
    
    result['f1_macro'] = f1_mac
    result['recall_macro'] = re_mac
    result['precision_macro'] = pre_mac
    
    print('Result: ', result)

    return result


def main(args):
    train_set = read_list_from_jsonl_file(args.train_path)
    val_set = read_list_from_jsonl_file(args.val_path)
    test_set = read_list_from_jsonl_file(args.test_path)
    
    if (args.mode == 'train'):
        run = wandb.init(
            # Set the project where this run will be logged
            project= "Biased Depression Model (BERT) on " + ("BDISen" if "BDISen" in args.test_path else "DepressionEmo"),
            name = current_time,
            # Track hyperparameters and run metadata
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "test_batch_size": args.test_batch_size,
                "max_length": args.max_length
            },
        )
        if not os.path.exists(args.resume_path):
            os.makedirs(args.resume_path)    
        train_model(train_set, val_set, pretrained_model = args.model_name, max_len = args.max_length, batch_size = args.batch_size,
                     saved_model_file = args.resume_path + '/best_bert_model.bin',
                     saved_history_file = args.resume_path + '/best_bert_model.json',
                     saved_best_threshold = args.resume_path + '/best_threshold.json', epochs = args.epochs, q_loss = args.q)
        wandb.finish()
        
    test_dataset(test_set, pretrained_model = args.model_name,
                    saved_model_file = args.resume_path + '/best_bert_model.bin',
                    saved_history_file = args.resume_path + '/best_bert_model.json',
                    saved_best_threshold = args.resume_path + '/best_threshold.json',
                    max_len = args.max_length, batch_size = args.test_batch_size)


if __name__ == "__main__":

    #fix random seed
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    current_time = time.strftime("%Y%m%d%H%M%S")

    parser = argparse.ArgumentParser(description='Training Parameters')
    parser.add_argument('--mode', type=str, default='train') # or test
    parser.add_argument('--model_name', type=str, default='bert-base-cased') # or test
    parser.add_argument('--train_path', type=str, default='dataset/train.json') 
    parser.add_argument('--test_path', type=str, default='dataset/test.json')
    parser.add_argument('--val_path', type=str, default='dataset/val.json')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--model_path', type=str, default='bart-base\checkpoint-452')
    parser.add_argument('--test_file', type=str, default='dataset/test.json')
    parser.add_argument('--q', type=float, default=0.5) #GCE parameter

    parser.add_argument('--resume_path', type=str, default='bert-base-debiased/'+current_time)
    args = parser.parse_args()
    
    print('args: ', args)

    label_list = symptom_list if "BDISen" in args.test_path else emotion_list

    main(args)
    
# python bert.py  --mode "train" --model_name "/data1/lipengfei/basemodels/bert-base-uncased" --epochs 25 --batch_size 8 --max_length 256 --train_path "Dataset/train.json" --val_path "Dataset/val.json" --test_path "Dataset/test.json"

# python bert.py --mode "test" --model_name "/data1/lipengfei/basemodels/bert-base-uncased" --train_path "Dataset/train.json" --val_path "Dataset/val.json" --test_path "Dataset/test.json" --max_length 256 --test_batch_size 16 --best_x -0.5 --best_y 0.0 --resume_path "bert-base/20241121203151"
