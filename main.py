
import matplotlib.pyplot as plt  
import os.path as os
from os import listdir

import numpy as np  
from hpelm import ELM

from skimage.feature import local_binary_pattern
from skimage import data as da
from skimage.color import label2rgb
from skimage.feature import local_binary_pattern
from skimage.segmentation import slic,mark_boundaries

import skimage as sk
import skimage.io as io

from yllib import *

# random((m, n), max) => m*n matrix
# random(n, max) => n*n matrix
random = lambda shape,maxx:(np.random.random(
shape if (isinstance(shape,tuple) or isinstance(shape,list)
)else (shape,shape))*maxx).astype(int)


imgDir = '../DUT-OMRON-image/'
imgResoultDir = '../pixelwiseGT-new-PNG/'

def readImg(imgName='im010.jpg',tag=0):
    ''' imgName名字
        tag 模式 为 1 只返还一张 否则 都返还
    '''
    img = io.imread(imgDir+imgName)
    if tag==1:
        return img
    
    img2 = io.imread(imgResoultDir+imgName[:-3]+'png')
    img2 = img2 != 0
    return img,img2




def show(l):
    if not isinstance(l,list):
        l = [l]
    n = len(l)
    fig, axes = plt.subplots(ncols=n)
    count = 0
    axes = [axes] if n==1 else axes 
    for img in l:
        axes[count].imshow(img)
        count += 1
#show([da.astronaut(),da.camera()])
        
DATA_NUM = 300 # test num


N_SEGMENTS = 200  # SLIC num
COMPACTNESS = 20  # SLIC compactiness

POS_MAX = int(N_SEGMENTS**0.5*5) # relative position

NN = 20 # ELM nerve num

def getLbp(img):
    '''返回 58 维lbp
    '''
    METHOD = 'uniform'
    RADIUS = 3  # LBP radius
    n_points = 56
    lbp = local_binary_pattern(img, n_points, RADIUS, METHOD)
    return lbp.astype(int)

#img = da.astronaut()
#gray = sk.color.rgb2gray(img)
#lbp = getLbp(gray)
#print lbp.max()
@performance
def getSlic(img):
    label = slic(img,N_SEGMENTS,COMPACTNESS)
    #show(mark_boundaries(img, label))
    return label
    

@performance
def getVectors(img, img2=None):
    
    labelMap = getSlic(img)
    
    gray = sk.color.rgb2gray(img)
    lbp = getLbp(gray)
    lbpLen = np.unique(lbp)
    
    lab = sk.color.rgb2lab(img).astype(int)
    
    labels = np.unique(labelMap)
    
    m ,n ,_= img.shape
    vectors = []
    answer = []
    for label in labels:
        superPixel = labelMap==label
        numSp = superPixel.sum()
        lbpHis = [0]*len(lbpLen)
        for i in lbp[superPixel]:
            lbpHis[i] += 1
            
        # 归一化
        lbpHis = map(lambda x:int(500*float(x)/numSp) , lbpHis)
        
        
        x=(np.vstack((np.arange(n),)*m)*superPixel).sum()//numSp
        y=(np.hstack((np.transpose([np.arange(m)]),)*n)*superPixel).sum()//numSp
        pos = [int(POS_MAX*x/n), int(POS_MAX*y/m)]
        
        L = lab[...,0][superPixel].sum()/numSp
        a = lab[...,1][superPixel].sum()/numSp
        b = lab[...,2][superPixel].sum()/numSp
        
        Lab = [L, a, b]
        vector = np.array([Lab+pos+lbpHis])
        vectors = np.vstack((vectors,vector)) if vectors!=[] else np.array(vector)
        
        if img2 != None:        
            yes=img2[superPixel].sum()/float(numSp)
            answer+=[(yes,1-yes)]    
    
    if img2 == None:
        return vectors

    return vectors,np.array(answer),labelMap

        




files = listdir(imgDir)
print 'files=',len(files)


from myEML import *
elm = []

for i in range(DATA_NUM):
    img,img2 = readImg(files[i])
    data,label,_ = getVectors(img,img2)
#    label = [[1,0] if index==0 else[0,1] for index in label]
    print '%d/%d training:%s' % (i,DATA_NUM,files[i])
    if elm == []:
#        w = np.array([1]*2)  ,classification='wc',w=w
        elm = getElm(data,label,nn=NN)
    else:
        elm.add_data(data,label)
        


def pre(begin=DATA_NUM,num=None):
    '''begin: begin of index
    num How much to predict
    '''
    if num == None:
        num = len(files) - begin
    resoult=None
    for i in range(begin,begin+num):
        name = files[i]
        img,img2 = readImg(name)
        data,label,_ = getVectors(img,img2)
        r=predict(elm,data,label)
        r=list(r[:2])        
        r[1]=len(r[1])
        resoult = [resoult[0]*resoult[1]+r[0]*r[1],resoult[1]+r[1]]\
        if resoult != None else [r[1]*r[0],r[1]]
        resoult[0]/=resoult[1]
        print (str(i)+' '+files[i]+' '*20)[:20],r
    print 'predict:',resoult



def preImg(name=DATA_NUM, color=False):
    '''name: filename or index in files
    to predict and show both raw img and resoult img in Ipython
    '''
    if isinstance(name,int):
        name = files[name]
    
    img,img2 = readImg(name)
    data,answer,labelMap = getVectors(img,img2)
    rate,ind,pt=predict(elm,data,answer)
    resoultImg = np.array(map(lambda row:
        map(lambda it:pt[it][0] if color else pt[it][0]>0.5,row)
        ,labelMap))
        
    
    show([img,mark_boundaries(img,labelMap)])
    show([img2,resoultImg])

    print 'resoult',rate

#pre(DATA_NUM,10)
preImg('im103.jpg')


