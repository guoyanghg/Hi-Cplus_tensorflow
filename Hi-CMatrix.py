
import numpy as np
import gzip
import matplotlib.pyplot as plt

HiC_max_value = 100

def recombine(matrix_size, data, padding):
    row_combine=[]
    result = []
    data_size = data.shape[-1]
    for j in range(matrix_size):
        for i in range(matrix_size):
            if i==0:
                row_combine = data[j*matrix_size+i][0][padding:data_size-padding , padding:data_size-padding]
            else:
                row_combine = np.column_stack((row_combine,data[j*matrix_size+i][0][padding:data_size-padding , padding:data_size-padding]))
        if j==0:
            result = row_combine
        else:
            result = np.row_stack((result, row_combine))
    return result

down_sample_ratio = 16
low_resolution_samples = np.load(gzip.GzipFile('./data/GM12878_replicate_down16_chr19_22.npy.gz', "r")) * down_sample_ratio
high_resolution_samples = np.load(gzip.GzipFile('./data/GM12878_replicate_original_chr19_22.npy.gz', "r"))
enhanced_samples = np.load('./data/enhanced_GM12878_replicate_down16_chr19_22_zuozhe.npy', "r")
enhanced_samples = enhanced_samples.reshape([14689,1,28,28])
print('low')
print(low_resolution_samples[0][0])

print('high')
print(high_resolution_samples[0][0])

print('enhanced')
print(enhanced_samples.shape)

Y = []
for i in range(high_resolution_samples.shape[0]):
    no_padding_sample = high_resolution_samples[i][0][6:34 , 6:34]
    Y.append(no_padding_sample)
Y = np.array(Y).astype(np.float32)
print('divide')
print(Y)


low_resolution_samples = np.minimum(HiC_max_value, low_resolution_samples)
high_resolution_samples = np.minimum(HiC_max_value, high_resolution_samples)
enhanced_samples = np.minimum(HiC_max_value, enhanced_samples)

r=recombine(121,low_resolution_samples,6)
print(r.shape)

plt.figure(1)
plt.imshow(r,cmap='Reds',origin="lower")
plt.suptitle("low_resolution")

r2=recombine(121,high_resolution_samples,6)
print(r2.shape)

plt.figure(2)
plt.imshow(r2,cmap='Reds',origin="lower")
plt.suptitle("high_resolution")

r3=recombine(121,enhanced_samples,0)
print(r3.shape)


plt.figure(3)
plt.imshow(r3,cmap='Reds',origin="lower")
plt.suptitle("enhanced_theano")
plt.show()

'''a=np.ones([500,1])
b=np.arange(1,100,1)*600
c=a*b
print(c)
plt.imshow(c,cmap='Reds')
plt.show()'''




