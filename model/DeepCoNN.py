# -*- coding: utf-8 -*-
# @Time    : 2020-05-15 21:11
# @Author  : zxl
# @FileName: DeepCoNN.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from gensim.models import word2vec,KeyedVectors
from Util.ConfigLoader import get_config

class Net(nn.Module):



    def __init__(self):
        super(Net,self).__init__()
        config=get_config('./config/model.yml')
        model = KeyedVectors.load_word2vec_format(
            './data/GoogleNews-vectors-negative300.bin', binary=True)
        weights = torch.FloatTensor(model.vectors).to("cuda:0")
        self.embedding = nn.Embedding.from_pretrained(weights).cuda()
        self.embedding.weight.requires_grad=False

        self.conv_u=nn.Sequential(nn.Conv1d(in_channels=config.embedding_size,
                                    out_channels=config.feature_size,
                                    kernel_size=config.window_size),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=config.max_text_len-config.window_size+1)).cuda()
        self.conv_i = nn.Sequential(nn.Conv1d(in_channels=config.embedding_size,
                                             out_channels=config.feature_size,
                                             kernel_size=config.window_size),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=config.max_text_len - config.window_size + 1)).cuda()

        self.fu=nn.Linear(in_features=config.feature_size,
                                        out_features=config.xu).cuda()
        self.fi=nn.Linear(in_features=config.feature_size,
                                         out_features=config.yi).cuda()

        self.f_fm=nn.Linear(in_features=config.xu+config.yi,
                            out_features=1,bias=True).cuda()
        self.relu=nn.ReLU()

        self.v=nn.Parameter(torch.randn(config.xu+config.yi,config.k),requires_grad=True).cuda()



    def fm_layer(self,x):

        linear_part=self.f_fm(x)
        inter_part1=torch.mm(x,self.v)
        inter_part1=torch.pow(inter_part1,2)
        inter_part2=torch.mm(torch.pow(x,2),torch.pow(self.v,2))
        output=linear_part+torch.sum(0.5*inter_part2-inter_part1)


        return output


    def forward(self,x):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        Xu=x[:,0].to(device)
        Xi=x[:,1].to(device)

        embedding_u=self.embedding(Xu)
        # print(embedding_u)
        embedding_u=embedding_u.permute(0,2,1)
        xu=self.conv_u(embedding_u)
        xu=xu.view(-1,xu.size(1))
        xu=self.fu(xu)

        embedding_i=self.embedding(Xi)
        embedding_i=embedding_i.permute(0,2,1)
        xi=self.conv_i(embedding_i)
        xi=xi.view(-1,xi.size(1))
        xi=self.fi(xi)

        cat_x=torch.cat((xu,xi),1)
        cat_x=self.relu(cat_x)

        output=self.fm_layer(cat_x)

        return output

class DeepCoNN():

    def __init__(self,conf):
        self.config=conf


    def fit(self,X,y):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net = Net()
        self.net.to(device)

        # for param in self.net.parameters():
        #     print(param.device)

        criterion=nn.MSELoss(reduction='sum')

        optimizer=optim.Adam(self.net.parameters(),lr=self.config.learning_rate)

        trainset=Data.TensorDataset(torch.from_numpy(X),torch.from_numpy(y))
        trainloader=Data.DataLoader(trainset,batch_size=self.config.batch_size,
                                                shuffle=True)
        best_loss = 100000000.0
        running_loss = 0.0
        for epoch in range(self.config.epoch):
            running_loss=0.0
            cur_loss=0.0
            for i, data in enumerate(trainloader,0):
                inputs,labels=data
                inputs=inputs.to(device)
                labels=labels.to(device)
                optimizer.zero_grad()

                outputs=self.net(inputs)

                loss=criterion(outputs.float(),labels.float())
                loss=loss.to(dtype=torch.float64)
                loss.backward()
                optimizer.step()
                running_loss+=loss.item()
                cur_loss+=loss.item()
                if i % 50 == 49:  # print every 2000 mini-batches
                    print('[%d,%d] loss: %.5f' %
                          (epoch + 1,i, float(cur_loss)/(self.config.batch_size*50)))
                    cur_loss=0.0
            print('[%d] loss: %.5f' %
                  (epoch + 1, float(running_loss)/len(X) ))
            if epoch % 5 == 4:
                if best_loss > running_loss/len(X):
                    best_loss = running_loss/len(X)
                    torch.save(self.net.state_dict(), self.config.save_path)
                    print("---------save best model!---------")
        if best_loss > running_loss/len(X):
            torch.save(self.net.state_dict(),self.config.save_path)

    def predict(self,X):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        net=Net()
        net.load_state_dict(torch.load(self.config.save_path))
        testset = torch.from_numpy(X)
        testloader = Data.DataLoader(testset, batch_size=self.config.batch_size,
                                      shuffle=False)
        pred_lst=[]
        with torch.no_grad():
            for i, data in enumerate(testloader,0):
                inputs=data.to(device)
                cur_res=net(inputs).cpu().numpy()
                pred_lst.extend(list(cur_res))

        return np.array(pred_lst)





