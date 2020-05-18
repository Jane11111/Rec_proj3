# DeepCoNN

This is the pytorch implementation for the paper:

Lei Zheng, Vahid Noroozi, and Philip S Yu. 2017. Joint deep modeling of users and items using reviews for recommendation. In WSDM. ACM, 425-434.


## Environment

python 3.7

pytorch 1.5.0

## Dataset

dataset: Amazon Digital_Music_5.json

pretrained word embedding: GoogleNews-vectors-negative300.bin


## How to run the code

### prepare data

put the data into ./data

``` 

 python preclean.py 

```

### train model

``` 
python train.py 

```

### predict 

``` 
python predict.py 

```
