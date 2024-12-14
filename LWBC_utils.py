import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader

import numpy as np

emotion_list = ['anger', 'brain dysfunction (forget)', 'emptiness', 'hopelessness', 'loneliness', 'sadness', 'suicide intent', 'worthlessness']
symptom_list = ['Sadness', 'Pessimism', 'Sense_of_failure', 'Loss_of_Pleasure', 'Guilty_feelings', 'Sense_of_punishment', 'Self-dislike', 'Self-incrimination', 'Suicidal_ideas', 'Crying', 'Agitation', 'Social_withdrawal', 'Indecision', 'Feelings_of_worthlessness', 'Loss_of_energy', 'Change_of_sleep', 'Irritability', 'Changes_in_appetite', 'Concentration_difficulty', 'Tiredness_or_fatigue', 'Loss_of_interest_in_sex']

class PreparedDataset():
    def __init__(self, texts, categories, tokenizer, max_len, num_classifier, set_size, nlbl):
        self.texts = texts
        self.categories = categories
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 初始化分类器掩码，用于生成采样集
        masks = np.array([[]] * num_classifier).tolist()
        for i in range(num_classifier):
            # 随机选择样本索引并为每个分类器生成一个采样掩码
            masks[i].append(np.random.choice(np.arange(len(self.texts)), set_size * nlbl))

        # 将掩码堆叠为一个数组，并初始化一个布尔掩码数组
        self.masks = np.stack(masks).reshape(num_classifier, -1)
        self.masks_place = torch.zeros(num_classifier, len(self.texts)).bool()
        for i in range(num_classifier):
            self.masks_place[i, self.masks[i]] = True   # 将对应索引位置设置为 True
        
        self.num_classifier = num_classifier
        self.encodings = []
        for text in self.texts:
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
            self.encodings.append(encoding)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        category = self.categories[item]
        encoding = self.encodings[item]
        mask = self.masks_place[:, item].tolist()       

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'categories': torch.tensor(category, dtype=torch.float)
        }

def create_data_loader(dataset, tokenizer, label_names, max_len, batch_size, num_classifier, set_size):

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
                         max_len=max_len, num_classifier=num_classifier,
                         set_size=set_size, nlbl=len(label_names))
    return DataLoader(ds, batch_size=batch_size, num_workers=4)


class CategoryClassifier(nn.Module):
    def __init__(self, n_labels, bert_model):
        super(CategoryClassifier, self).__init__()
        self.bert = bert_model
        self.drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, n_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, return_dict=False):
        with torch.no_grad():
            _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        output = self.drop(pooled_output)
        output = self.out(output)
        output = self.sigmoid(output)
        return output


class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def ce(outputs, labels, reduce=True):
    if reduce:
        return (- labels * torch.log(outputs)).sum(dim=1)
    return - labels * torch.log(outputs)

