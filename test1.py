import os
import wfdb as wf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
path='D:/data/'
filenamelist= list(os.listdir(path))
print (filenamelist)


signal=[]
label=[]
for file in filenamelist:
    if '.atr' in file:
        annotation=wf.rdann(file)  #读取注释
        print(annotation)
        label.append()
    if '.dat' in file:
        signals,fields=wf.rdsamp(file)
        record=wf.rdrecord(file)
        print(record)
'''
#Load AFDB


id_list=['04015','04048','04126','04746','04908','05121','05261','06426','06453','06995','07162','07859','07879','07910','08215','08219','08378','08405','08434','08455']
print(len(id_list))
data1=[]
data2=[]
label=[]
for id in id_list:
        signal,fields=wf.rdsamp(id,channels=[0,1])
        #print(signal.shape)
        signal1 = signal[:, 0]
        signal2 = signal[:, 1]

        #print('signal:',signal)
        #print('fields:',fields)

        annotation = wf.rdann(id, extension='atr')
        #print('annotation:', annotation.__dict__)
        #sample=annotation.__dict__['sample']
        sample = annotation.__dict__['sample'].tolist()
        sample.append(signal.shape[0])
        #print(sample)
        aux_note=annotation.__dict__['aux_note']
        label.extend(aux_note)
        #print(label)
        for i in range(len(sample) - 1):
                data1.append(signal1[sample[i]:sample[i + 1]])
                data2.append(signal1[sample[i]:sample[i + 1]])
                # print(data)




print(len(label))
print(len(data1))
print(len(data2))
Nofnumber=label.count('(N')
AFofnumber=label.count('(AFIB')
Jofnumber=label.count('(J')
AFLofnumber=label.count('(AFL')
#print(Nofnumber,AFofnumber,Jofnumber,AFLofnumber)
print(label)

#### one hot 编码
import keras
for index,value in enumerate(label):
        if value=='(N':
                label[index]=0
        if value=='(AFIB':
                label[index]=1
        if value=='(J':
                label[index]=2
        if value=='(AFL':
                label[index]=3

label= keras.utils.to_categorical(label)
print(label)

# 存储为mat文件
import scipy.io
scipy.io.savemat('ecg1.mat',mdict={'data1':data1})
scipy.io.savemat('ecg2.mat',mdict={'data2':data2})
scipy.io.savemat('label.mat',mdict={'label':label})




# 数据分段
def data_slice(data, label, length):
    slice_data = []
    slice_label=[]
    n = len(data)
    for i in range(n):
        m = len(data[i])//length
        #data=data[i].tolist()
        for j in range(1,m+1):
            slice_data.append(data[(j-1)*length:j*length])
        slice_label.append(m*label[i])
    return slice_data,slice_label


length = 10000
slice, label = data_slice(data1, label, length)
print(len(slice))
