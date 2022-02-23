
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
gpu_id = '0,1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
os.system('echo $CUDA_VISIBLE_DEVICES')
config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction =1
config.gpu_options.allow_growth = True
from keras.backend.tensorflow_backend import set_session
set_session(tf.Session(config=config))
from model import classfier
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
matplotlib.use('Agg')

def onehot(y):
    one_hot = np.eye(4).tolist()
    train_y = list(np.array(y).squeeze())
    label_y = []
    for i in range(len(train_y)):
        j = train_y[i]
        label_y.append(one_hot[j])
    return np.array(label_y)
def train(nn_param,x_train,  y_train,x_test, y_test):
    tf.reset_default_graph()
    multi = classfier(architecture=nn_param)
    acc=multi.fit(x_train, y_train,x_test, y_test)
    return acc,multi
def cross_valadition(x,y,nn_param,k_size = 10):
    print("cross valadition --------------------")
    k_fold = StratifiedKFold(k_size, True, random_state=len(y))
    index = k_fold.split(X=x, y=np.argmax(y,axis=1))
    y_pred_list=[]
    y_true_list=[]
    for train_index, test_index in index:
        x_train = (x[train_index])
        x_test = (x[test_index])
        y_train = y[train_index]
        y_test = (y[test_index])
        acc, model = train(nn_param, x_train,  y_train,x_test, y_test)
        predict_y, pred = model.predict(x_test, y_test)
        y_pred_list.append(pred)
        y_true_list.append(y_test)
    return y_pred_list,y_true_list

def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]
def get_diffential_gene(gene1,gene2):
    wt = gene1.mean(axis=0)
    ko = gene2.mean(axis=0)
    fold =ko - wt
    pvalue = []
    for i in range(gene1.shape[1]):
        ttest = stats.ttest_ind(gene1[:, i], gene2[:, i])
        pvalue.append(ttest[1])
    qvalue = p_adjust_bh(np.asarray(pvalue))
    fold_cutoff = 1
    qvalue_cutoff = 0.05
    filtered_ids = list()
    for i in range(gene1.shape[1]):
        if(abs(fold[i]) >= fold_cutoff) and (qvalue[i] <= qvalue_cutoff):
             filtered_ids.append(i)
    return filtered_ids

def load_data(all_label_class,path,labelpath):#):
    data=list()
    label=list()
    label_indext=list()
    non_consense_num=0
    nolbl_sum=0
    gene_name = list()
    #with open(r'{}.tsv'.format(path), 'r') as f:
    if 'TCGA' in path:
        with open(path,
                  'r') as f:
            for line in f:
               data.append(line.strip('\n').split('\t'))
    else:
        with open(path, 'r') as f:
            for line in f:
               data.append(line.strip('\n').split('\t'))
    gene_set = data
    sample=gene_set[0]
    if 'TCGA' in path:
        del sample[0]#tcga datasets
    del gene_set[0]
    for i in range(len(gene_set)):
        gene_name.append(gene_set[i][0])
        del gene_set[i][0]
    for i in range(len(sample)):
          temp = sample[i]

          if '.' in temp:
              temp = sample[i][:sample[i].index('.')]
              sample[i] = temp
          if '_' in temp:
              temp=sample[i][:sample[i].index('_')]
              sample[i]=temp
          label.append([])
    #print(sample)
    #print(len(sample))
    #data.clear()
    with open(labelpath,'r') as f:
        line = f.readline()
        print(line.split('\t')[14])
        for line in f:
            temp = line.split('\t')[0]
            if temp in sample:
                index = sample.index(temp)
                # label[index]=line.split('\t')[2]
                label[index] = line.split('\t')[14]

    one_hot=np.eye(len(all_label_class)).tolist()
    label_all=list()
    #for i in range(len(label)):
    geneset_array = np.array(gene_set,dtype=float).T


    i=0
    #print(len(label))
    #print(len(geneset_array))

    while i<len(label):
        if label[i] not in all_label_class:
            if len(label[i])==0:
                non_consense_num+=1
            else:
                nolbl_sum+=1
            del label[i]

            geneset_array=np.delete(geneset_array,i,0)
            i=i-1
        i=i+1
    for i in range(len(label)):
        j=all_label_class.index(label[i])
        label_all.append(one_hot[j])
        label_indext.append(j)

    CMS1 = np.where(np.array(label_indext)==0)
    CMS2 = np.where(np.array(label_indext) == 1)
    CMS3 = np.where(np.array(label_indext)==2)
    CMS4 = np.where(np.array(label_indext) == 3)

    CMS1_features= np.squeeze(geneset_array[CMS1,:])
    CMS2_features = np.squeeze(geneset_array[CMS2, :])
    CMS3_features = np.squeeze(geneset_array[CMS3, :])
    CMS4_features = np.squeeze(geneset_array[CMS4, :])
    diff_gene=list()
    diff_gene.append(get_diffential_gene(CMS1_features,CMS2_features))
    diff_gene.append(get_diffential_gene(CMS1_features, CMS3_features))
    diff_gene.append(get_diffential_gene(CMS1_features, CMS4_features))
    diff_gene.append(get_diffential_gene(CMS2_features, CMS3_features))
    diff_gene.append(get_diffential_gene(CMS2_features, CMS4_features))
    diff_gene.append(get_diffential_gene(CMS3_features, CMS4_features))
    diff_gene_index=list()
    for i in diff_gene:
        for j in i:
            if j not in diff_gene_index:
                diff_gene_index.append(j)

    gene_name=np.array(gene_name)[np.array(diff_gene_index)]

    return np.array(label_all), geneset_array[:, np.array(diff_gene_index)], geneset_array, np.array(label_indext), gene_name, sample

def main():
    all_label_class = ['CMS1', 'CMS2', 'CMS3', 'CMS4']
    datasetname = ["GSE13067","GSE13294","GSE14333"
                   ,"GSE17536","GSE20916","GSE2109"
                   ,"GSE37892","GSE39582"]
    for name in datasetname:
        path=r'/data/'+name+'.tsv'
        y, x, a, b, gene_name,sample = load_data(all_label_class,path)
        nn_param = {'nb_neurons': [768, 128], 'dropout_rate': [0.2, 0.1],
                    'l1_rate': [0.0001, 0.01], 'l2_rate': [0.01, 0], 'BN': [0, 0],
                    'activation': ['relu', 'relu'], 'loss': 'categorical_crossentropy',
                    'optim': 'Adam', 'learning_rate': 0.0001, 'batch_size': 256, 'epoch': 500, 'nb_layers': 2}
        fit, model = cross_valadition(x, y, nn_param)


if __name__ == "__main__":
    main()









