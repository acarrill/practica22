#!/usr/bin/python2.7
#-*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

B=8 # Tamano del bloque
Imagen= 'zelda.bmp'
img1 = cv2.imread(Imagen,cv2.CV_LOAD_IMAGE_UNCHANGED)
h,w=np.array(img1.shape[:2])/B * B
img1=img1[:h,:w]
#Convert BGR to RGB
b,g,r = cv2.split(img1)
img2 = cv2.merge((r,g,b))
plt.figure()
plt.imshow(img2)

point=plt.ginput(1)
block=np.floor(np.array(point)/B) #first component is col, second component is row
print "Coordinates of selected block: ",block
scol=block[0,0]
srow=block[0,1]
plt.plot([B*scol,B*scol+B,B*scol+B,B*scol,B*scol],[B*srow,B*srow,B*srow+B,B*srow+B,B*srow])
plt.axis([0,w,h,0])

transcol=cv2.cvtColor(img1, cv2.cv.CV_BGR2YCrCb)
SSV=2
SSH=2
crf=cv2.boxFilter(transcol[:,:,1],ddepth=-1,ksize=(2,2))
cbf=cv2.boxFilter(transcol[:,:,2],ddepth=-1,ksize=(2,2))
crsub=crf[::SSV,::SSH]
cbsub=cbf[::SSV,::SSH]
imSub=[transcol[:,:,0],crsub,cbsub]

QY=np.array([[16,11,10,16,24,40,51,61],
[12,12,14,19,26,48,60,55],[14,13,16,24,40,57,69,56],
[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],
[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],
[72,92,95,98,112,100,103,99]])
QC=np.array([[17,18,24,47,99,99,99,99],
[18,21,26,66,99,99,99,99],[24,26,56,99,99,99,99,99],
[47,66,99,99,99,99,99,99],[99,99,99,99,99,99,99,99],
[99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99],
[99,99,99,99,99,99,99,99]])

QF=80.0
if QF < 50 and QF > 1:
    scale = np.floor(5000/QF)
elif QF < 100:
    scale = 200-2*QF
else:
    print "Quality Factor must be in the range [1..99]"
scale=scale/100.0
Q=[QY*scale,QC*scale,QC*scale]

TransAll=[]
TransAllQuant=[]
ch=['Y','Cr','Cb']
plt.figure()
for idx,channel in enumerate(imSub):
    plt.subplot(1,3,idx+1)
    channelrows=channel.shape[0]
    channelcols=channel.shape[1]
    Trans = np.zeros((channelrows,channelcols), 
                     np.float32)
    TransQuant = np.zeros((channelrows,channelcols), 
                          np.float32)
    blocksV=channelrows/B
    blocksH=channelcols/B
    vis0 = np.zeros((channelrows,channelcols), 
                    np.float32)
    vis0[:channelrows, :channelcols] = channel
    vis0=vis0-128
    for row in range(blocksV):
        for col in range(blocksH):
            currentblock = cv2.dct(vis0[row*B:
                (row+1)*B,col*B:(col+1)*B])
            Trans[row*B:(row+1)*B,col*B:
                (col+1)*B]=currentblock
            TransQuant[row*B:(row+1)*B,col*B:
                (col+1)*B]=np.round(currentblock/Q[idx])
    TransAll.append(Trans)
    TransAllQuant.append(TransQuant)
    if idx==0:
        selectedTrans=Trans[srow*B:(srow+1)*B,scol*B:
            (scol+1)*B]
    else:
        sr=np.floor(srow/SSV)
        sc=np.floor(scol/SSV)
        selectedTrans=Trans[sr*B:(sr+1)*B,sc*B:(sc+1)*B]
    
    plt.imshow(selectedTrans,cmap=cm.jet,interpolation='nearest')
    plt.colorbar(shrink=0.5)
    
    
    
ImagDecodificada=np.zeros((h,w,3), np.uint8)
for idx,channel in enumerate(TransAllQuant):
    channelrows=channel.shape[0]
    channelcols=channel.shape[1]
    blocksV=channelrows/B
    blocksH=channelcols/B
    back0 = np.zeros((channelrows,channelcols), np.uint8)
    for row in range(blocksV):
        for col in range(blocksH):
            dequantblock=channel[row*B:(row+1)*B,col*B:(col+1)*B]*Q[idx]
            currentblock = np.round(cv2.idct(dequantblock))+128
            currentblock[currentblock>255]=255
            currentblock[currentblock<0]=0
            back0[row*B:(row+1)*B,col*B:
                        (col+1)*B]=currentblock
        back1=cv2.resize(back0,(w,h))
        ImagDecodificada[:,:,idx]=np.round(back1)    

reImg=cv2.cvtColor(ImagDecodificada, cv2.cv.CV_YCrCb2BGR)
cv2.cv.SaveImage('BackTransformedQuant.jpg', 
                 cv2.cv.fromarray(reImg))
plt.figure()
img3=np.zeros(img1.shape,np.uint8)
img3[:,:,0]=reImg[:,:,2]
img3[:,:,1]=reImg[:,:,1]
img3[:,:,2]=reImg[:,:,0]
plt.imshow(img3)
SSE=np.sqrt(np.sum((img2-img3)**2))
print "Sum of squared error: ",SSE