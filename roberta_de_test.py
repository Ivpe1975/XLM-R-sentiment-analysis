import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import torch
import transformers
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
model_name = "xlm-roberta-base"


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


# df = pd.read_json("json/train/dataset_de_train.json", orient="records", lines=True)
# df = df.loc[df['stars']!=3]
# df["sentiment"] = df["stars"].apply(lambda x: 0 if x < 3 else 1)
# df = df[['review_body','sentiment']]
# df["review_body"] = df["review_body"].apply(lambda x: x if isinstance(x,str) else " ")


# Defining some key variables that will be used later on in the training
MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
# EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, do_lower_case=True)

class SentimentData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.review_body
        self.targets = self.data.sentiment
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
# df_de = pd.read_json("json/test/dataset_de_test.json", orient="records", lines=True)
# df_de = df_de.loc[df_de['stars']!=3]
# df_de["sentiment"] = df_de["stars"].apply(lambda x: 0 if x < 3 else 1)
# df_de = df_de[['review_body','sentiment']]
# df_de["review_body"] = df_de["review_body"].apply(lambda x: x if isinstance(x,str) else " ")

# df_fr = pd.read_json("json/test/dataset_fr_test.json", orient="records", lines=True)
# df_fr = df_fr.loc[df_fr['stars']!=3]
# df_fr["sentiment"] = df_fr["stars"].apply(lambda x: 0 if x < 3 else 1)
# df_fr = df_fr[['review_body','sentiment']]
# df_fr["review_body"] = df_fr["review_body"].apply(lambda x: x if isinstance(x,str) else " ")

# df_ja = pd.read_json("json/test/dataset_ja_test.json", orient="records", lines=True)
# df_ja = df_ja.loc[df_ja['stars']!=3]
# df_ja["sentiment"] = df_ja["stars"].apply(lambda x: 0 if x < 3 else 1)
# df_ja = df_ja[['review_body','sentiment']]
# df_ja["review_body"] = df_ja["review_body"].apply(lambda x: x if isinstance(x,str) else " ")

df_en = pd.read_json("json/test/dataset_en_test.json", orient="records", lines=True)
df_en = df_en.loc[df_en['stars']!=3]
df_en["sentiment"] = df_en["stars"].apply(lambda x: 0 if x < 3 else 1)
df_en = df_en[['review_body','sentiment']]
df_en["review_body"] = df_en["review_body"].apply(lambda x: x if isinstance(x,str) else " ")

# df_es = pd.read_json("json/test/dataset_es_test.json", orient="records", lines=True)
# df_es = df_es.loc[df_es['stars']!=3]
# df_es["sentiment"] = df_es["stars"].apply(lambda x: 0 if x < 3 else 1)
# df_es = df_es[['review_body','sentiment']]
# df_es["review_body"] = df_es["review_body"].apply(lambda x: x if isinstance(x,str) else " ")
#
# df_zh = pd.read_json("json/test/dataset_zh_test.json", orient="records", lines=True)
# df_zh = df_zh.loc[df_zh['stars']!=3]
# df_zh["sentiment"] = df_zh["stars"].apply(lambda x: 0 if x < 3 else 1)
# df_zh = df_zh[['review_body','sentiment']]
# df_zh["review_body"] = df_zh["review_body"].apply(lambda x: x if isinstance(x,str) else " ")


# test_data_ja = df_ja.reset_index(drop=True)
# test_data_fr = df_fr.reset_index(drop=True)
# test_data_de = df_de.reset_index(drop=True)
test_data_en = df_en.reset_index(drop=True)
# test_data_es = df_es.reset_index(drop=True)
# test_data_zh = df_zh.reset_index(drop=True)

# train_data = df.reset_index(drop=True)


# training_set = SentimentData(train_data, tokenizer, MAX_LEN)
# testing_set_de = SentimentData(test_data_de, tokenizer, MAX_LEN)
# testing_set_ja = SentimentData(test_data_ja, tokenizer, MAX_LEN)
# testing_set_fr = SentimentData(test_data_fr, tokenizer, MAX_LEN)
testing_set_en = SentimentData(test_data_en, tokenizer, MAX_LEN)
# testing_set_es = SentimentData(test_data_es, tokenizer, MAX_LEN)
# testing_set_zh = SentimentData(test_data_zh, tokenizer, MAX_LEN)


# train_params = {'batch_size': TRAIN_BATCH_SIZE,
#                 'shuffle': False,
#                 'num_workers': 0
#                 }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

# training_loader = DataLoader(training_set, **train_params)

# testing_loader_de = DataLoader(testing_set_de, **test_params)
# testing_loader_ja = DataLoader(testing_set_ja, **test_params)
# testing_loader_fr = DataLoader(testing_set_fr, **test_params)
testing_loader_en = DataLoader(testing_set_en, **test_params)
# testing_loader_es = DataLoader(testing_set_es, **test_params)
# testing_loader_zh = DataLoader(testing_set_zh, **test_params)

class NN_class(torch.nn.Module):
    def __init__(self):
        super(NN_class, self).__init__()
        self.l1 = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768,2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


model = torch.load('pytorch_roberta_en.bin')
model.to(device)
# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


def calcuate_accuracy(preds, targets):

    n_correct = (preds==targets).sum().item()
    return n_correct

def calcuate_tp(preds,targets):
    tp=0
    for i,j in zip(preds,targets):
        if i==j and i==1:
            tp+=1
    return tp

def calcuate_fp(preds,targets):
    fp=0
    for i,j in zip(preds,targets):
        if i==1 and j==0:
            fp+=1
    return fp

def calcuate_fn(preds,targets):
    fn=0
    for i,j in zip(preds,targets):
        if i==0 and j==1:
            fn+=1
    return fn

def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    return

# EPOCHS = 1
# for epoch in range(EPOCHS):
#     train(epoch)


def valid(model, testing_loader):
    model.eval()
    y_pred=[]
    y_true=[]
    n_correct = 0; n_wrong = 0; total = 0; tr_loss=0; nb_tr_steps=0; nb_tr_examples=0
    tp=0
    fp=0
    fn=0
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accuracy(big_idx, targets)
            for i in big_idx:
                y_pred.append(i.item())
            for i in targets:
                y_true.append(i.item())
            tp+=calcuate_tp(big_idx, targets)
            fp+=calcuate_fp(big_idx, targets)
            fn+=calcuate_fn(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)

    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    precision=(tp)/(tp+fp)
    recall=(tp)/(tp+fn)
    f1_score=(2*tp)/(2*tp+fp+fn)

    return f1_score,precision,recall,epoch_accu,y_pred,y_true



# f1_score_de, precision_de, recall_de, epoch_accu_de = valid(model, testing_loader_de)
# f1_score_ja, precision_ja, recall_ja, epoch_accu_ja = valid(model, testing_loader_ja)
# f1_score_fr, precision_fr, recall_fr, epoch_accu_fr = valid(model, testing_loader_fr)
f1_score_en, precision_en, recall_en, epoch_accu_en, y_pred, y_true = valid(model, testing_loader_en)
# f1_score_es, precision_es, recall_es, epoch_accu_es = valid(model, testing_loader_es)
# f1_score_zh, precision_zh, recall_zh, epoch_accu_zh = valid(model, testing_loader_zh)

wrong_ids=[]
for idx,(pred,true) in enumerate(zip(y_pred,y_true)):
    if pred!=true:
        wrong_ids.append(idx)
print(wrong_ids)
with open('y_pred.txt','w+') as pred:
    for idx, i in enumerate(y_pred):
        if idx!=len(y_pred)-1:
            pred.write(str(i)+',')
        else:
            pred.write(str(i))
with open('y_true.txt','w+') as pred:
    for idx, i in enumerate(y_true):
        if idx!=len(y_true)-1:
            pred.write(str(i)+',')
        else:
            pred.write(str(i))
# with open('results_de.txt','w+') as results:
#     results.write('f_1 for German from German: ' + str(f1_score_de)+'\n')
#     results.write('f_1 for Japanese from German: ' + str(f1_score_ja)+'\n')
#     results.write('f_1 for French from German:' + str(f1_score_fr)+'\n')
#     results.write('f_1 for English from German:' + str(f1_score_en)+ '\n')
#     results.write('f_1 for Spanish from German:' + str(f1_score_es)+ '\n')
#     results.write('f_1 for Chinese from German:' + str(f1_score_zh)+ '\n')
#
#     results.write('Precision for German from German: ' + str(precision_de)+'\n')
#     results.write('Precision for Japanese from German: ' + str(precision_ja)+'\n')
#     results.write('Precision for French from German:' + str(precision_fr)+'\n')
#     results.write('Precision for English from German:' + str(precision_en)+ '\n')
#     results.write('Precision for Spanish from German:' + str(precision_es)+ '\n')
#     results.write('Precision for Chinese from German:' + str(precision_zh)+ '\n')
#
#     results.write('Recall for German from German: ' + str(recall_de)+'\n')
#     results.write('Recall for Japanese from German: ' + str(recall_ja)+'\n')
#     results.write('Recall for French from German:' + str(recall_fr)+'\n')
#     results.write('Recall for English from German:' + str(recall_en)+ '\n')
#     results.write('Recall for Spanish from German:' + str(recall_es)+ '\n')
#     results.write('Recall for Chinese from German:' + str(recall_zh)+ '\n')
#
#     results.write('Accuracy for German from German: ' + str(epoch_accu_de)+'\n')
#     results.write('Accuracy for Japanese from German: ' + str(epoch_accu_ja)+'\n')
#     results.write('Accuracy for French from German:' + str(epoch_accu_fr)+'\n')
#     results.write('Accuracy for English from German:' + str(epoch_accu_en)+ '\n')
#     results.write('Accuracy for Spanish from German:' + str(epoch_accu_es)+ '\n')
#     results.write('Accuracy for Chinese from German:' + str(epoch_accu_zh)+ '\n')


# output_model_file = 'pytorch_roberta_de.bin'
# output_vocab_file = 'de_vocab/'

# model_to_save = model
# torch.save(model_to_save, output_model_file)
# tokenizer.save_vocabulary(output_vocab_file)
