import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import wandb
import argparse
from collections import defaultdict
from tqdm import tqdm
from torch.utils import data

import matplotlib.pyplot as plt
import collections
from tqdm import tqdm
import warnings
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from file_io import read_list_from_jsonl_file, write_single_dict_to_json_file
warnings.filterwarnings("ignore")

np.set_printoptions(3, suppress=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
current_time = time.strftime("%Y%m%d%H%M%S")

from LWBC_utils import *


parser = argparse.ArgumentParser()

### COMMON
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--max_length', type=int, default=256)
parser.add_argument('--temperature', type=float, default=200)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--dataset', default='emotion', type=str, help='emotion, BDISen')

parser.add_argument('--local', dest='local', action='store_true', help='disable wandb')

parser.add_argument('--alpha', default=0.02, type=float, help='alpha')

### bootstrap
parser.add_argument('--warmup', type=int, default=20)
parser.add_argument('--loss_scale', type=float, default=0.2)
parser.add_argument('--num_classifier', type=int, default=5)
parser.add_argument('--set_size', type=int, default=40, help='total_size = set_size * num_class')
parser.add_argument('--warmup_init', dest='warmup_init', type=int, default=0)

parser.add_argument('--linear_bias', dest='linear_bias', action='store_true', help='linear_bias')

### KD
parser.add_argument('--kd_lambda', type=float, default=0.6)
parser.add_argument('--kd_temperature', type=float, default=1)


### path
parser.add_argument('--save_path', type=str, default='bert-base-LWBC/'+current_time)
parser.add_argument('--data_path', type=str, default='Dataset')


### COMMONS
parser.set_defaults(local=False)
config = parser.parse_args()

pretrained_model = "/data1/lipengfei/basemodels/bert-base-uncased"

label_list = symptom_list if config.dataset == "BDISen" else emotion_list
metrics_hisrtory_path = config.save_path + '/metrics.json'
if not os.path.exists(config.save_path):
    os.makedirs(config.save_path)
    os.makedirs(config.save_path+'/checkpoints')
    with open(metrics_hisrtory_path, 'w+') as f:
        pass

seed = config.seed
# seed = 101
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)

if not config.local:
    wandb.init(
        config=config,
        project='LBC_{}'.format(config.dataset),
        anonymous='allow',
    )

if config.dataset == "BDISen":
    train_path = config.data_path + '/train_BDISen.json'
    val_path = config.data_path + '/val_BDISen.json'
    test_path = config.data_path + '/test_BDISen.json'
else:
    train_path = config.data_path + '/train.json'
    val_path = config.data_path + '/val.json'
    test_path = config.data_path + '/test.json'


train_set = read_list_from_jsonl_file(train_path)
val_set = read_list_from_jsonl_file(val_path)
test_set = read_list_from_jsonl_file(test_path)
tokenizer = BertTokenizer.from_pretrained(pretrained_model)
train_data_loader = create_data_loader(train_set, tokenizer, label_list, config.max_length, config.batch_size, config.num_classifier, config.set_size)
val_data_loader = create_data_loader(val_set, tokenizer, label_list, config.max_length, config.batch_size, config.num_classifier, config.set_size)
test_data_loader = create_data_loader(test_set, tokenizer, label_list, config.max_length, config.batch_size, config.num_classifier, config.set_size)

f_dim = 512
output_dim = len(label_list)
num_classifier = config.num_classifier
classifiers = []

bert_model = BertModel.from_pretrained(pretrained_model)
for _ in range(num_classifier):
    classifiers.append(CategoryClassifier(output_dim, bert_model).to(device))
target_classifier = CategoryClassifier(output_dim, bert_model).to(device)

## SINGLE optimizer
params = []
for i in range(num_classifier):
    params += list(classifiers[i].parameters())

if config.optimizer =='SGD':
    print("SGD")
    optimizer=torch.optim.SGD(params, lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    optimizer_target=torch.optim.SGD(target_classifier.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
else:
    optimizer=torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    optimizer_target=torch.optim.Adam(target_classifier.parameters(), lr=config.lr, weight_decay=config.weight_decay)

kl_loss = torch.nn.KLDivLoss(reduce=False).to(device)

best_metric = {}
best_metric['val_macro_f1'] = 0 
best_metric['val_micro_f1'] = 0 
cur_threshold = [0.5 for _ in label_list]
cur_threshold = torch.tensor(cur_threshold).to(device)

for epoch in range(config.epochs):
    print('_' * 50)
    print("Epoch: ", epoch+1)
    losses = AverageMeter()
    target_losses = AverageMeter()
    kd_losses = AverageMeter()
    ens_losses = AverageMeter()
    for i in range(num_classifier):
        classifiers[i] = classifiers[i].train()
    target_classifier = target_classifier.train()
    weights = []
    num_samples = 0
    tot_correct = torch.Tensor([[0] * len(label_list)]*(num_classifier+1)).to(device)
    tot_preds = [[] for _ in range(num_classifier+1)]
    tot_true = [[] for _ in range(num_classifier+1)]
    for d in tqdm(train_data_loader, leave=False, desc='Train Epoch {}'.format(epoch+1)):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        mask = d["mask"].to(device)
        categories = d["categories"].to(device)
        optimizer.zero_grad()
        optimizer_target.zero_grad()
        loss = 0 

        num_samples += input_ids.shape[0]
        outputs =[]
        loss_cand = []
        for i in range(num_classifier):
            outputs.append(classifiers[i](input_ids=input_ids, attention_mask=attention_mask)*config.temperature)  
            preds = (outputs[i]/config.temperature > cur_threshold).to(torch.float)              
            tot_correct[i] += (preds == categories).sum(dim=0)
            tot_preds[i].extend(preds.cpu().numpy().tolist())
            tot_true[i].extend(categories.cpu().numpy().tolist())
            count = mask[:,i].sum()
            if count>0:
                loss += (ce(outputs[i], categories, True)*mask[:,i].to(device)).sum()/mask[:,i].sum()

        if epoch+1 > config.warmup:
            target_outputs = target_classifier(input_ids=input_ids, attention_mask=attention_mask)*config.temperature
            target_preds = (target_outputs/config.temperature > cur_threshold).to(torch.float)
            tot_correct[-1] += (target_preds == categories).sum(dim=0)
            tot_preds[-1].extend(target_preds.cpu().numpy().tolist())
            tot_true[-1].extend(categories.cpu().numpy().tolist())
            weight = (1/(torch.stack([outputs[i].detach() for i in range(num_classifier)]).float().sum(dim=0)/num_classifier+config.alpha))
            weights.extend(weight.detach().cpu().numpy().tolist())
            target_loss = (ce(target_outputs, categories, False)*weight.detach()).sum(dim=1).mean()*config.loss_scale
            
            target_losses.update(target_loss.item())
            target_loss.backward()
            optimizer_target.step()

            target_outputs = target_classifier(input_ids=input_ids, attention_mask=attention_mask)*config.temperature
            if config.kd_lambda > 0 and epoch > config.warmup:
                kl_outputs = F.softmax(target_outputs.detach() / config.kd_temperature)
                loss = loss * (1-config.kd_lambda)
                ens_losses.update(loss)
                for i in range(num_classifier):
                    kd_loss_ = kl_loss(F.log_softmax(outputs[i] / config.kd_temperature), kl_outputs) * (~mask[:,i][..., None].to(device))
                    kd_loss = config.kd_lambda * kd_loss_.mean() * (config.kd_temperature ** 2) 
                    
                    loss += kd_loss
                    kd_losses.update(kd_loss.item())
                    
            else:
                ens_losses.update(loss)
        if type(loss) is not int: # all masked
            loss.backward()
            optimizer.step()
            losses.update(loss.item())
            mean_weight = np.mean(weights)

    train_avg_accuracy = [tot_correct[k] / num_samples for k in range(num_classifier+1)]
    train_f1_macro = [f1_score(tot_true[k], tot_preds[k], average='macro') for k in range(num_classifier+1)]
    train_f1_micro = [f1_score(tot_true[k], tot_preds[k], average='micro') for k in range(num_classifier+1)]
          
    metrics = {
        'train_loss': losses.avg,
        'train_target_loss': target_losses.avg,
        'weights': mean_weight,
        'kd_loss': kd_losses.avg*num_classifier,
        'ens_loss': ens_losses.avg/num_classifier
    }

    # Eval
    # if epoch+1 > config.warmup:
    if epoch+1 > -1:
        num_samples = 0
        tot_correct = torch.Tensor([[0] * len(label_list)]*(num_classifier+1)).to(device)
        tot_preds = [[] for _ in range(num_classifier+1)]
        tot_true = [[] for _ in range(num_classifier+1)]
        for d in tqdm(val_data_loader, leave=False, desc='Val Epoch {}'.format(epoch+1)):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            mask = d["mask"].to(device)
            categories = d["categories"].to(device)
            batch_size = input_ids.shape[0]
            num_samples += batch_size
            for k, classifier in enumerate(classifiers):
                classifier = classifier.eval()
                with torch.no_grad():
                    outputs = classifier(input_ids=input_ids, attention_mask=attention_mask)
                preds = (outputs > cur_threshold).to(torch.float)
                tot_correct[k] += (preds == categories).sum(dim=0)
                tot_preds[k].extend(preds.cpu().numpy().tolist())
                tot_true[k].extend(categories.cpu().numpy().tolist())
            target_classifier = target_classifier.eval()
            with torch.no_grad():
                outputs = target_classifier(input_ids=input_ids, attention_mask=attention_mask)
            preds = (outputs > cur_threshold).to(torch.float)
            tot_correct[-1] += (preds == categories).sum(dim=0)
            tot_preds[-1].extend(preds.cpu().numpy().tolist())
            tot_true[-1].extend(categories.cpu().numpy().tolist())

        valid_avg_accuracy = [tot_correct[k] / num_samples for k in range(num_classifier+1)]
        valid_f1_macro = [f1_score(tot_true[k], tot_preds[k], average='macro') for k in range(num_classifier+1)]
        valid_f1_micro = [f1_score(tot_true[k], tot_preds[k], average='micro') for k in range(num_classifier+1)]
                
        print(f"Epoch {epoch+1} Train loss: {losses.avg} target_loss: {target_losses.avg} train_acc {train_avg_accuracy[-1]} val_acc {valid_avg_accuracy[-1]}")
        print(f"Train F1 macro: {train_f1_macro[-1]} micro: {train_f1_micro[-1]}")
        
        metrics['train_acc'] = train_avg_accuracy[-1].tolist()
        metrics['train_macro_f1'] = train_f1_macro[-1]
        metrics['train_micro_f1'] = train_f1_micro[-1]

        metrics['val_acc'] = valid_avg_accuracy[-1].tolist()
        metrics['val_macro_f1'] = valid_f1_macro[-1]
        metrics['val_micro_f1'] = valid_f1_micro[-1]
        
        for k in range(num_classifier):
            metrics['train_acc_M{}'.format(k)]=train_avg_accuracy[k].tolist()
            metrics['val_acc_M{}'.format(k)]=valid_avg_accuracy[k].tolist()
        
        if best_metric['val_macro_f1'] + best_metric['val_micro_f1'] < valid_f1_macro[-1] + valid_f1_micro[-1]:
            best_metric = metrics.copy()
            best_metric['epoch'] = epoch+1
            torch.save(target_classifier.state_dict(), '{}/checkpoints/best_target.bin'.format(config.save_path))
            for cls_number in range(num_classifier):
                torch.save(classifiers[cls_number].state_dict(), '{}/checkpoints/best_classifier_{}.bin'.format(config.save_path, cls_number))
        
    else:
        print("Epoch {} Train_loss: ".format(epoch+1), losses.avg)


    write_single_dict_to_json_file(metrics_hisrtory_path, metrics)

    if not config.local:
        wandb.log(metrics)

# Read best model
target_classifier.load_state_dict(torch.load('{}/checkpoints/best_target.bin'.format(config.save_path)))
for cls_number in range(num_classifier):
    classifiers[cls_number].load_state_dict(torch.load('{}/checkpoints/best_classifier_{}.bin'.format(config.save_path, cls_number)))


num_samples = 0
tot_correct = torch.Tensor([[0] * len(label_list)]*(num_classifier+1)).to(device)
tot_preds = [[] for _ in range(num_classifier+1)]
tot_true = [[] for _ in range(num_classifier+1)]
for d in tqdm(test_data_loader, leave=False, desc='Test'):
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    mask = d["mask"].to(device)
    categories = d["categories"].to(device)
    batch_size = input_ids.shape[0]
    num_samples += batch_size
    for k, classifier in enumerate(classifiers):
        classifier.eval()
        with torch.no_grad():
            outputs = classifier(input_ids=input_ids, attention_mask=attention_mask)
        preds = (outputs > cur_threshold).to(torch.float)
        tot_correct[k] += (preds == categories).sum(dim=0)
        tot_preds[k].extend(preds.cpu().numpy().tolist())
        tot_true[k].extend(categories.cpu().numpy().tolist())
    target_classifier.eval()
    with torch.no_grad():
        outputs = target_classifier(input_ids=input_ids, attention_mask=attention_mask)
    preds = (outputs > cur_threshold).to(torch.float)
    tot_correct[-1] += (preds == categories).sum(dim=0)
    tot_preds[-1].extend(preds.cpu().numpy().tolist())
    tot_true[-1].extend(categories.cpu().numpy().tolist())

test_avg_accuracy = [tot_correct[k] / num_samples for k in range(num_classifier+1)]
test_f1_macro = [f1_score(tot_true[k], tot_preds[k], average='macro') for k in range(num_classifier+1)]
test_f1_micro = [f1_score(tot_true[k], tot_preds[k], average='micro') for k in range(num_classifier+1)]


print("FINISH")
print(f"best epoch {best_metric['epoch']} val_macro_f1 {best_metric['val_macro_f1']} val_micro_f1 {best_metric['val_micro_f1']}")
print(f"Test acc {test_avg_accuracy[-1]}")
print(f"Test F1 macro: {test_f1_macro[-1]} micro: {test_f1_micro[-1]}")

if not config.local:
    wandb.finish()