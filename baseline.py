#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# In[5]:


# Dev Data
data_de, data_en, data_es, data_fr, data_ja, data_zh = 0,0,0,0,0,0
_lang_ = ["de","en","es","fr","ja","zh"]

for i in range(len(_lang_)):
    df = pd.read_json(f"json/train/dataset_{_lang_[i]}_train.json", orient="records", lines=True)
    df = df.loc[df['stars']!=3]
    df["sentiment"] = df["stars"].apply(lambda x: 0 if x < 3 else 1)
    df = df[['review_body','sentiment']]
    df["review_body"] = df["review_body"].apply(lambda x: x if isinstance(x,str) else " ")

    if i == 0:
        combine = df
    else:
        temp_list = []
        temp_list.append(combine)
        temp_list.append(df)
        combine = pd.concat(temp_list)
        data = combine



# In[6]:


# Test Data
df = pd.read_json(f"json/test/dataset_de_test.json", orient="records", lines=True)
df = df.loc[df['stars']!=3]
df["sentiment"] = df["stars"].apply(lambda x: 0 if x < 3 else 1)
df = df[['review_body','sentiment']]
df["review_body"] = df["review_body"].apply(lambda x: x if isinstance(x,str) else " ")
test_data_de = df


df = pd.read_json(f"json/test/dataset_en_test.json", orient="records", lines=True)
df = df.loc[df['stars']!=3]
df["sentiment"] = df["stars"].apply(lambda x: 0 if x < 3 else 1)
df = df[['review_body','sentiment']]
df["review_body"] = df["review_body"].apply(lambda x: x if isinstance(x,str) else " ")
test_data_en = df


df = pd.read_json(f"json/test/dataset_es_test.json", orient="records", lines=True)
df = df.loc[df['stars']!=3]
df["sentiment"] = df["stars"].apply(lambda x: 0 if x < 3 else 1)
df = df[['review_body','sentiment']]
df["review_body"] = df["review_body"].apply(lambda x: x if isinstance(x,str) else " ")
test_data_es = df


df = pd.read_json(f"json/test/dataset_fr_test.json", orient="records", lines=True)
df = df.loc[df['stars']!=3]
df["sentiment"] = df["stars"].apply(lambda x: 0 if x < 3 else 1)
df = df[['review_body','sentiment']]
df["review_body"] = df["review_body"].apply(lambda x: x if isinstance(x,str) else " ")
test_data_fr = df


df = pd.read_json(f"json/test/dataset_ja_test.json", orient="records", lines=True)
df = df.loc[df['stars']!=3]
df["sentiment"] = df["stars"].apply(lambda x: 0 if x < 3 else 1)
df = df[['review_body','sentiment']]
df["review_body"] = df["review_body"].apply(lambda x: x if isinstance(x,str) else " ")
test_data_ja = df


df = pd.read_json(f"json/test/dataset_zh_test.json", orient="records", lines=True)
df = df.loc[df['stars']!=3]
df["sentiment"] = df["stars"].apply(lambda x: 0 if x < 3 else 1)
df = df[['review_body','sentiment']]
df["review_body"] = df["review_body"].apply(lambda x: x if isinstance(x,str) else " ")
test_data_zh = df


# In[9]:


X = data["review_body"]
y = data["sentiment"]
vec = CountVectorizer(min_df=0.0005)
x = vec.fit_transform(X).toarray()
print(x.shape)


x_test_de = test_data_de["review_body"]
y_test_de = test_data_de["sentiment"]
x_test_de = vec.transform(x_test_de).toarray()

x_test_en = test_data_en["review_body"]
y_test_en = test_data_en["sentiment"]
x_test_en = vec.transform(x_test_en).toarray()

x_test_es = test_data_es["review_body"]
y_test_es = test_data_es["sentiment"]
x_test_es = vec.transform(x_test_es).toarray()

x_test_fr = test_data_fr["review_body"]
y_test_fr = test_data_fr["sentiment"]
x_test_fr = vec.transform(x_test_fr).toarray()

x_test_ja = test_data_ja["review_body"]
y_test_ja = test_data_ja["sentiment"]
x_test_ja = vec.transform(x_test_ja).toarray()

x_test_zh = test_data_zh["review_body"]
y_test_zh = test_data_zh["sentiment"]
x_test_zh = vec.transform(x_test_zh).toarray()


# In[10]:

model = MultinomialNB()
model.fit(x, y)


# In[34]:

ground_truth = [y_test_de, y_test_en, y_test_es, y_test_fr, y_test_ja, y_test_zh]
predictions = [model.predict(x_test_de), model.predict(x_test_en), model.predict(x_test_es), model.predict(x_test_fr),
               model.predict(x_test_ja), model.predict(x_test_zh)]


# In[43]:

file = open("bseline_metrics.txt", "w")

for i in range(len(ground_truth)):
    file.write(f"{_lang_[i]} Acc: {accuracy_score(ground_truth[i], predictions[i])}\n")
    file.write(f"{_lang_[i]} F1: {f1_score(ground_truth[i], predictions[i])}\n")

file.close()
