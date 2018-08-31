import numpy as np
from creat_data import creat_cnf
from compute_acc import acc_compute
import tensorflow as tf
from time import  time
from model import make_action

size_of_X=20
size_of_C=60
k_sat=3

CHECKFILE = './checkpoint/model.ckpt'
lr=1e-5

try_step=30
n_epoch=100
def train():

    make_action(size_of_X,size_of_C,k_sat,try_step,n_epoch,is_Training=1)


train()