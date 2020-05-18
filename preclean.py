# -*- coding: utf-8 -*-
# @Time    : 2020-05-17 11:55
# @Author  : zxl
# @FileName: preclean.py

import json
import numpy as np
import pandas as pd
import nltk.tokenize as tk
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors


def save_data(file_path,train_path,test_path,val_path):
   f = open(file_path, 'r')
   l = f.readline()
   reviewID_lst = []
   asin_lst = []
   reviewText_lst = []
   overall_lst = []

   while l:
      json_obj = json.loads(l)
      reviewID_lst.append(json_obj['reviewerID'])
      asin_lst.append(json_obj['asin'])
      reviewText_lst.append(json_obj['reviewText'])
      overall_lst.append(json_obj['overall'])

      l = f.readline()

   df = pd.DataFrame(
      {'reviewerID': reviewID_lst, 'asin': asin_lst, 'reviewText': reviewText_lst, 'overall': overall_lst})

   trainset, testset = train_test_split(df[['reviewerID', 'asin', 'reviewText', 'overall']], test_size=0.2,
                                        shuffle=True)
   testset, valset = train_test_split(testset[['reviewerID', 'asin', 'reviewText', 'overall']], test_size=0.5,
                                      shuffle=True)

   trainset.to_csv(train_path, sep='\t', index=False, header=True)
   testset.to_csv(test_path, sep='\t', index=False, header=True)
   valset.to_csv(val_path, sep='\t', index=False, header=True)

   train_user = set(trainset.reviewerID.values)
   test_user = set(testset.reviewerID.values)
   val_user = set(valset.reviewerID.values)
   train_item = set(trainset.asin.values)
   test_item = set(testset.asin.values)
   val_item = set(valset.asin.values)

   print('train user: %d, test_user: %d, val_user: %d' % (len(train_user), len(test_user), len(val_user)))
   print('train item: %d, test_item: %d, val_item: %d' % (len(train_item), len(test_item), len(val_item)))
   print('train & test: %d, train & val: %d' % (
   len(train_user.intersection(test_user)), len(train_user.intersection(val_user))))
   print('train & test: %d, train & val: %d' % (
   len(train_item.intersection(test_item)), len(train_item.intersection(val_item))))


def fit_data(df,vocab_dic):
   tokenizer = tk.WordPunctTokenizer()
   max_count=0
   user_review_dic={}
   item_review_dic={}
   for user,group in df.groupby(['reviewerID']):
       user_review_dic[user]=set([])
       cur_count=0
       doc =' '.join( group['reviewText'].values)
       tokens = tokenizer.tokenize(doc)
       for token in tokens:
           if token in vocab_dic:
               if vocab_dic[token] not in user_review_dic[user]:
                   user_review_dic[user].add(vocab_dic[token])
                   cur_count+=1
       max_count=max(cur_count,max_count)



   for item,group in df.groupby(['asin']):
       item_review_dic[item]=set([])
       doc =' '.join( group['reviewText'].values)
       cur_count=0
       tokens = tokenizer.tokenize(doc)
       for token in tokens:
           if token in vocab_dic:
               if vocab_dic[token] not in item_review_dic[item]:
                   item_review_dic[item].add(vocab_dic[token])
                   cur_count+=1
       max_count=max(cur_count,max_count)

   return  user_review_dic,item_review_dic,max_count

def transform_data(df,user_review_dic,item_review_dic,total_count):
    y_lst=[]
    X_lst=[]
    for reviewerID,asin,reviewText,overall in zip(df.reviewerID,df.asin,df.reviewText,df.overall):
        u_lst=list(user_review_dic[reviewerID])
        i_lst=list(item_review_dic[asin])
        if len(u_lst)>total_count:
            u_lst=u_lst[:total_count]
        if len(i_lst)>total_count:
            i_lst=i_lst[:total_count]
        while len(u_lst)<total_count:
            u_lst.append(0)
        while len(i_lst)<total_count:
            i_lst.append(0)
        X_lst.append([u_lst,i_lst])
        y_lst.append([overall])
    return X_lst,y_lst


if __name__ == "__main__":

   file_path="./data/Digital_Music_5.json"
   train_path='./data/train.csv'
   test_path='./data/test.csv'
   val_path='./data/val.csv'
   train_X_path='./data/trainX.npy'
   train_y_path='./data/trainy.npy'
   test_X_path='./data/testX.npy'
   test_y_path='./data/testy.npy'
   val_X_path='./data/valX.npy'
   val_y_path='./data/valy.npy'
   config_path='./config/model.yml'


   #save_data(file_path,train_path,test_path,val_path)
   train_df=pd.read_csv(train_path,sep='\t')
   test_df=pd.read_csv(test_path,sep='\t')
   val_df=pd.read_csv(val_path,sep='\t')

   model = KeyedVectors.load_word2vec_format(
       './data/GoogleNews-vectors-negative300.bin', binary=True)

   print('GoogleNews-vectors-negative300 loaded!')
   vocab_dic={model.index2word[i]:i for i in range(len(model.index2word))}
   user_review_dic, item_review_dic, max_count = fit_data(train_df,vocab_dic)
   max_count=3000
   print('max text length: %d'%max_count)
   print('data fitting finished!')
   X_train,y_train=transform_data(train_df,user_review_dic,item_review_dic,max_count)
   print('train transforming finished!')
   X_val,y_val=transform_data(val_df,user_review_dic,item_review_dic,max_count)
   print('validation transforming finished!')
   X_test, y_test = transform_data(test_df, user_review_dic, item_review_dic, max_count)
   print('test transforming finished!')
   print('start saving.....')
   np.save(train_X_path,np.array(X_train))
   np.save(train_y_path,np.array(y_train))
   np.save(val_X_path,np.array(X_val))
   np.save(val_y_path,np.array(y_val))
   np.save(test_X_path,np.array(X_test))
   np.save(test_y_path,np.array(y_test))
   print('saving finished!')

   print(np.array(X_test).shape)
   print(np.array(y_test).shape)

   arr1=np.load(test_X_path)
   print(arr1.shape)
   arr2=np.load(test_y_path)
   print(arr2.shape)








