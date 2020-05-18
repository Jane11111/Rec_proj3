# -*- coding: utf-8 -*-
# @Time    : 2020-05-16 18:41
# @Author  : zxl
# @FileName: main.py

import numpy as np
import torch
from Util.ConfigLoader import get_config
from model.DeepCoNN import DeepCoNN

if __name__ == "__main__":


   train_X_path='./data/trainX.npy'
   train_y_path='./data/trainy.npy'
   test_X_path='./data/testX.npy'
   test_y_path='./data/testy.npy'
   val_X_path='./data/valX.npy'
   val_y_path='./data/valy.npy'
   config_path='./config/model.yml'
   res_path='./data/res.npy'

   train_X=np.load(train_X_path)
   train_y=np.load(train_y_path)
   test_X=np.load(test_X_path)
   test_y=np.load(test_y_path)
   print(train_X.shape)
   print(train_y.shape)
   print(test_X.shape)
   print(test_y.shape)

   config = get_config(config_path)
   torch.cuda.set_device(0)

   if torch.cuda.is_available():
      device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
      print("Running on the GPU")
   else:
      device = torch.device("cpu")
      print("Running on the CPU")

   model = DeepCoNN(config)

   print('----start training----')


   model.fit(train_X, train_y)

   # print('----start testing-----')
   # res = model.predict(test_X)
   # # print(res)
   # mse=mean_squared_error(test_y,res)
   # print('mse: %f'%mse)
   #
   # np.save(res_path,res)


