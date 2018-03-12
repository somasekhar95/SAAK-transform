import torch
import argparse
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from data.datasets import MNIST
import torch.utils.data as data_utils
from sklearn.decomposition import PCA
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import product
from sklearn.decomposition import PCA
from sklearn import svm

# argument parsing
print(torch.__version__)
batch_size=1
test_batch_size=1
kwargs={}
train_loader=data_utils.DataLoader(MNIST(root='./data',train=True,process=False,transform=transforms.Compose([
    transforms.Scale((32,32)),
    transforms.ToTensor(),
])),batch_size=batch_size,shuffle=True,**kwargs)

train_labels = []

for x in train_loader:
    img, label = x


test_loader=data_utils.DataLoader(MNIST(root='./data',train=False,process=False,transform=transforms.Compose([
    transforms.Scale((32,32)),
    transforms.ToTensor(),
])),batch_size=test_batch_size,shuffle=True,**kwargs)



# show sample
def show_sample(inv):
    inv_img=inv.data.numpy()[0][0]
    plt.imshow(inv_img)
    plt.gray()
    plt.savefig('D:/Docs_required/EE669/hw4/Saak-Transform-master/image/demo.png')
   # plt.show()

'''
@ For demo use, only extracts the first 1000 samples
'''
def create_numpy_dataset():
    datasets = []
    for data in train_loader:
        data_numpy = data[0].numpy()
        data_numpy = np.squeeze(data_numpy)
        datasets.append(data_numpy)
    datasets = np.array(datasets)
    datasets=np.expand_dims(datasets,axis=1)
    print('Numpy dataset shape is {}'.format(datasets.shape))
    return datasets[:1000]



'''
@ data: flatten patch data: (14*14*60000,1,2,2)
@ return: augmented anchors
'''
def PCA_and_augment(data_in):
    # data reshape
    data=np.reshape(data_in,(data_in.shape[0],-1))
    print('PCA_and_augment: {}'.format(data.shape))
    # mean removal
    mean = np.mean(data, axis=0)
    datas_mean_remov = data - mean
    print('PCA_and_augment meanremove shape: {}'.format(datas_mean_remov.shape))

    # PCA, retain all components
    pca=PCA()
    pca.fit(datas_mean_remov)
    comps=pca.components_

    # augment, DC component doesn't
    comps_aug=[vec*(-1) for vec in comps[:-1]]
    comps_complete=np.vstack((comps,comps_aug))
    print('PCA_and_augment comps_complete shape: {}'.format(comps_complete.shape))
    return comps_complete



'''
@ datasets: numpy data as input
@ depth: determine shape, initial: 0
'''

def fit_pca_shape(datasets,depth):
    factor=np.power(2,depth)
    length=32/factor
    print('fit_pca_shape: length: {}'.format(length))
    idx1=range(0,int(length),2)
    idx2=[i+2 for i in idx1]
    print('fit_pca_shape: idx1: {}'.format(idx1))
    data_lattice=[datasets[:,:,i:j,k:l] for ((i,j),(k,l)) in product(zip(idx1,idx2),zip(idx1,idx2))]
    data_lattice=np.array(data_lattice)
    print('fit_pca_shape: data_lattice.shape: {}'.format(data_lattice.shape))

    #shape reshape
    data=np.reshape(data_lattice,(data_lattice.shape[0]*data_lattice.shape[1],data_lattice.shape[2],2,2))
    print('fit_pca_shape: reshape: {}'.format(data.shape))
    return data


'''
@ Prepare shape changes. 
@ return filters and datasets for convolution
@ aug_anchors: [7,4] -> [7,input_shape,2,2]
@ output_datas: [60000*num_patch*num_patch,channel,2,2]

'''
def ret_filt_patches(aug_anchors,input_channels):
    shape=aug_anchors.shape[1]/4
    num=aug_anchors.shape[0]
    filt=np.reshape(aug_anchors,(num,int(shape),4))

    # reshape to kernels, (7,shape,2,2)
    filters=np.reshape(filt,(num,int(shape),2,2))

    # reshape datasets, (60000*shape*shape,shape,28,28)
    # datasets=np.expand_dims(dataset,axis=1)

    return filters



'''
@ input: numpy kernel and data
@ output: conv+relu result
'''
def conv_and_relu(filters,datasets,stride=2):
    # torch data change
    filters_t=torch.from_numpy(filters)
    datasets_t=torch.from_numpy(datasets)

    # Variables
    filt=Variable(filters_t).type(torch.FloatTensor)
    data=Variable(datasets_t).type(torch.FloatTensor)

    # Convolution
    output=F.conv2d(data,filt,stride=stride)

    # Relu
    relu_output=F.relu(output)

    return relu_output,filt



'''
@ One-stage Saak transform
@ input: datasets [60000,channel, size,size]
'''
def one_stage_saak_trans(datasets=None,depth=0):


    # load dataset, (60000,1,32,32)
    # input_channel: 1->7
    print('one_stage_saak_trans: datasets.shape {}'.format(datasets.shape))
    input_channels=datasets.shape[1]

    # change data shape, (14*60000,4)
    data_flatten=fit_pca_shape(datasets,depth)

    # augmented components
    comps_complete=PCA_and_augment(data_flatten)
    print('one_stage_saak_trans: comps_complete: {}'.format(comps_complete.shape))

    # get filter and datas, (7,1,2,2) (60000,1,32,32)
    filters=ret_filt_patches(comps_complete,input_channels)
    print('one_stage_saak_trans: filters: {}'.format(filters.shape))

    # output (60000,7,14,14)
    relu_output,filt=conv_and_relu(filters,datasets,stride=2)

    data=relu_output.data.numpy()
    print('one_stage_saak_trans: output: {}'.format(data.shape))
    return data,filt,relu_output



'''
@ Multi-stage Saak transform
'''
def multi_stage_saak_trans():
    filters = []
    outputs = []

    data=create_numpy_dataset()
    dataset=data
    #print(dataset)
    num=0
    img_len=data.shape[-1]
    while(img_len>=2):
        num+=1
        img_len/=2


    for i in range(num):
        print ('{} stage of saak transform: '.format(i))
        data,filt,output=one_stage_saak_trans(data,depth=i)
        filters.append(filt)
        outputs.append(output)
        print ('')


    return dataset,filters,outputs

'''
@ Reconstruction from the second last stage
@ In fact, reconstruction can be done from any stage
'''
def toy_recon(outputs,filters):
    outputs=outputs[::-1][1:]
    filters=filters[::-1][1:]
    num=len(outputs)
    data=outputs[0]
    for i in range(num):
        data = F.conv_transpose2d(data, filters[i], stride=2)

    return data

def column(matrix, i):
 return [row[i] for row in matrix]
    

if __name__=='__main__':
 dataset,filters,outputs=multi_stage_saak_trans()
 '''print(filters)
 [f,p] = stats.f_oneway(outputs)
 print(f)
 '''
 '''
 test_datasets = [] check
 for data in train_loader:
    data_numpy = data[0].numpy()
    data_numpy = np.squeeze(data_numpy)
    test_datasets.append(data_numpy)
 test_datasets = np.array(test_datasets)
 test_datasets=np.expand_dims(test_datasets,axis=1)
 #print(test_datasets)
 '''
 
 import math
 div = 26 #check
 #test = test_datasets[1500:2500]
 #test = np.array(test)
 #test = test.flatten()
 #print(test.shape)
 #test = np.reshape(test,(1000,-1))
 #print('test shape')
 #print(test.shape)
 reduced_set_labels = []
 labels = []
 for data in train_loader:
    labels_numpy = data[1].numpy()
    labels_numpy = np.squeeze(labels_numpy)
    labels.append(labels_numpy)
 labels = np.array(labels)
 reduced_set_labels = labels[:1000]
 
 
 class_indi = [[0 for x in range(200)] for y in range(10)]
 class_indi = np.array(class_indi)
 features = []
 comp = 25
 #print('length of output features')
 #print(len(outputs[4][:,250]))

 #c = []
 #c = np.asarray(column(outputs[4],1998))
 
 #check = [[0 for x in range(1000)] for y in range(1500)]
 #check = np.array(check)
 
 #check[1][:]= outputs[4][:,250]
 #print('check')
 #print(check[1][:])
 #1500 x 1000 array

 num_of_samples = []
 coress_feature = []
 F_score = []
 count =0
 for i in range(1999): #1999
    BGV = 0.0
    WGV = 0.0
    error1 = 0.0
    features = column(outputs[4], i)
    features = np.array(features)
    sum = np.sum(features[0])
    mean_for_all_classes = sum/1000
    for j in range(10):
        class_sum=0.0
        inter_sum = 0.0
        indi = np.where(reduced_set_labels == j)
        #print('number of samples per class')
        #print((indi[0].shape)[0])
        num_of_samples.append((indi[0].shape)[0])
        indi = np.array(indi)
        error1 = 1-comp/div
        N = 200 - num_of_samples[j]
        indi = np.lib.pad(indi,(0,N),'constant')
        class_indi[j][:] = indi[0]
        error2 = 0.0
        for k in class_indi[j][:]:
            coress_feature  = features[k]
            if(k==(num_of_samples[j]-1)):
                break
        for nums in coress_feature:
            class_sum = class_sum + nums
        mean_of_each_class = class_sum/num_of_samples[j]    
        BGV = BGV + (num_of_samples[j]* pow((np.subtract(mean_for_all_classes,mean_of_each_class)),2)) 
        for nums in coress_feature:
            inter_sum = inter_sum + pow(np.subtract(nums,mean_of_each_class),2)
        WGV = WGV + inter_sum
    BGV = BGV/9    
    WGV = WGV/991
    count = count + 1
    comps = 34
    print('count :%d' %count)
    #print('BGV and WGV values:')
    #print(BGV)
    #print(WGV)
    #print(BGV/WGV)
    F_score.append(BGV/WGV)    
 
 F_score = np.array(F_score).tolist()
 print('F test score')
 print(F_score)
 sorted_index = []
 sorted_index = sorted(range(200), key=lambda k: F_score[k]) # 1999
 sorted_index = np.array(sorted_index).tolist()
 top_feature_index = sorted_index[499:1999]
 n_comps = 64
 divs = n_comps/2 + math.floor(n_comps/21)

 
 reduced_data = [[0 for x in range(1000)] for y in range(1500)] #1500
 #reduced_data = np.array(reduced_data)
 i=0
 error2_ = 0.0
 for cols in top_feature_index:
    print(cols)
    #print('data in cols')
    #print(outputs[4][:][cols])
    reduced_data[i][:] = outputs[4][:,cols]
    i = i + 1
    print(i)
 #print(reduced_data)
 error2 = 1-comps/divs
 reduced_data_arr = np.array(reduced_data)
 reduced_data_arr = reduced_data_arr.transpose()
 print(reduced_data_arr.shape)
 #print(reduced_data_arr)    
 
 #data_PCA = [[0 for x in range(1500)] for y in range(1000)]
 #data_PCA = reduced_data_arr
 #data_PCA = np.array(data_PCA)
 
 pca = PCA(n_components = 64)
 pca.fit(reduced_data_arr)
 data_64 = pca.transform(reduced_data_arr)
 
 
 train_data_64 = [[0 for x in range(64)] for y in range(750)]
 test_data_64 = [[0 for x in range(64)] for y in range(250)]
 
 error1_ = 0
 for i in range(750):
    train_data_64[i][:] = data_64[i,:]
 train_data_64 = np.array(train_data_64)
 print(train_data_64.shape)
 
 
 for i in range(250):
    test_data_64[i][:] = data_64[750+i,:]
 test_data_64 = np.array(test_data_64)
 print(test_data_64.shape)
 
 
 
reduced_set_train_labels = labels[:750]
 print(reduced_set_train_labels.shape)
 
 
 reduced_set_test_labels = []
 for i in range(250):
  reduced_set_test_labels.append(labels[:750+i])
 reduced_set_test_labels = np.array(reduced_set_test_labels)
 print(reduced_set_test_labels.shape)
 
 clf = svm.SVC(decision_function_shape='ovo')
 clf.fit(train_data_64, reduced_set_train_labels)
 
 training_pred = clf.predict(data_64)
 misclass_train = 0
 train_percent_acc = 0.0
 for i in range(750):
    if training_pred[i] != reduced_set_train_labels[i]:
        misclass_train += 1
 error1_ = misclass_train/1000
 train_percent_acc = (1-error1)*100
 print('traning accuracy :')
 print(train_percent_acc)
 
 testing_pred = clf.predict(test_data_64)
 print(testing_pred.shape)
 testing_pred = testing_pred.tolist()
 
 reduced_set_test_labels = reduced_set_test_labels.tolist()
 misclass = 0
 percent_acc = 0.0
 for i in range(250):
    if testing_pred[i] != reduced_set_test_labels[i]:
        misclass += 1
 error2_ = misclass/1000
 percent_acc = (1-error2)*100
 print('testing accuracy :')
 print(percent_acc)
 
 data=toy_recon(outputs,filters)
 show_sample(data)






