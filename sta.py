# -*- coding: utf-8 -*-
# @Time    : 2020-05-17 14:27
# @Author  : zxl
# @FileName: sta.py

import pandas as pd
import numpy as np
import nltk.tokenize as tk

tokenizer = tk.WordPunctTokenizer()


def sta(df):
    user_set=set(df.reviewerID.values)
    item_set=set(df.asin.values)
    review_count=0
    word_count=0
    word_set=set([])
    for user, group in df.groupby(['reviewerID']):
        review_count+=group.shape[0]
        # print(group['reviewText'].values)
        for reviewText in group['reviewText'].values:
            if type(reviewText)== str:
                word_lst=tokenizer.tokenize(reviewText)
                for word in word_lst:
                    word_count+=1
                    word_set.add(word)
    # word_count=len(word_set)

        # doc = ' '.join(group['reviewText'].values)
        # word_count+=len(tokenizer.tokenize(doc))


    print('user: %d, item: %d, review: %d, word: %d,review per user: %f, word per review: %f'%(
        len(user_set),len(item_set),review_count,word_count,review_count/len(user_set),word_count/review_count,
    ))


if __name__ == "__main__":
    train_path='./data/train.csv'
    test_path='./data/test.csv'
    val_path='./data/val.csv'

    train_df=pd.read_csv(train_path,sep='\t')
    test_df=pd.read_csv(test_path,sep='\t')
    val_df=pd.read_csv(val_path,sep='\t')
    df=pd.concat([train_df,test_df,val_df])
    sta(val_df)
