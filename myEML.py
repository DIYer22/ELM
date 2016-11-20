
import scipy.io as sio  
import matplotlib.pyplot as plt  
import numpy as np  
from hpelm import ELM


def getElm(data,label,clas=2,nn=10):

    print 'data,label shape=',data.shape,label.shape
    
    elm = ELM(data.shape[1], label.shape[1])
    elm.add_neurons(nn, "sigm")
    elm.train(data, label, "c")
    return elm    
    
    
    
def predict(elm, data, label):
    pt = elm.predict(data)
    ind = pt.argmax(axis=1)
    resoultMap = ind==label.argmax(axis=1)
    re = float(resoultMap.sum())/resoultMap.shape[0]
    return re,ind,pt




