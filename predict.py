# -*- coding: utf-8 -*-
# @Time    : 2020-05-17 16:03
# @Author  : zxl
# @FileName: predict.py

import numpy as np
from Util.ConfigLoader import get_config
from model.DeepCoNN import DeepCoNN
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":



   test_X_path='./data/testX.npy'
   test_y_path='./data/testy.npy'
   config_path='./config/model.yml'
   res_path='./data/res.npy'


   test_X=np.load(test_X_path)
   test_y=np.load(test_y_path)

   print(test_X.shape)
   print(test_y.shape)
   config = get_config(config_path)
   model = DeepCoNN(config)

   print('----start testing-----')
   res = model.predict(test_X)
   mse=mean_squared_error(test_y,res)
   print('mse: %f'%mse)

   np.save(res_path,res)


