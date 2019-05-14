from scipy import signal
import numpy as np

import wfdb as wf



data1=[]
data2=[]
label=[]
signal,fields=wf.rdsamp('04015',channels=[0,1])
#print(signal.shape)
signal1 = signal[:, 0]
signal2 = signal[:, 1]
annotation = wf.rdann('04015', extension='atr')
sample = annotation.__dict__['sample'].tolist()
sample.append(signal.shape[0])
#print(sample)
aux_note=annotation.__dict__['aux_note']
label.extend(aux_note)

for i in range(len(sample) - 1):
        data1.append(signal1[sample[i]:sample[i + 1]])
        data2.append(signal1[sample[i]:sample[i + 1]])
        # print(data)

print(data1[1].shape)
print(len(data1[1]))
#print((data1[1].tolist())


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








'''
# To spectrogram
def to_spectrogram(signal):
    _, _, Sxx = signal.spectrogram(signal, fs=300, window=('tukey', 0.25),
                                     nperseg=64, noverlap=0.5, return_onesided=True)

    return Sxx.T
def fouier_transformation(signals_whole):
    
    spectrogram = np.apply_along_axis(to_spectrogram, 1, signals_whole)

    # Log transformation and Standardizer
    log_spectrogram = np.log(spectrogram + 1)

    centers = log_spectrogram.mean(axis=(1,2))
    stds = log_spectrogram.std(axis=(1,2))
    log_spectrogram_s = np.array([(x - c) / d for x, c, d in zip(log_spectrogram, centers, stds)])
    x_s = log_spectrogram_s[..., np.newaxis]
    return x_s
'''