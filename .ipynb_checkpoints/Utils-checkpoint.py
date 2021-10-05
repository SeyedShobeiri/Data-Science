import os
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
proj_root_path = '/Users/homayoonshobeiri/Desktop/Python for Github/Data Science ShowCase for Recruiters/Data-Science/Utils'
images_path = os.path.join(proj_root_path,"Images")

def save_fig(fig_id,tight_layout=True,fig_extension="png",resolution=96):
    path = os.path.join(images_path,fig_id+"."+fig_extension)
    print("Saving figure",fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path,format=fig_extension,dpi=resolution)
    
    
def test_set_check(identifier,test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def train_test_maker(data,test_ratio,id = "index"):
    # In order to prevent the algo to see all of the data when 
    # this process is run we use the most stable feature (row ids) to build
    # unique identifier (make sure new data get's appended to the end of DF and 
    # no data ever gets deleted 
    # we can also use the comination of features to create an id
    ids = data[id]
    in_test_set = ids.apply(lambda id_: test_set_check(id_,test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


def stratum_maker(data,column,stratum_name,bins,labels):
    data[stratum_name] = pd.cut(data[column],bins,labels)
    return data[stratum_name]

def stratified_shuffled_sampling(data,column,stratum_name,bins,labels,n_splits,test_size,random_state=42):
    data[stratum_name] = pd.cut(data[column],bins,labels)
    split = StratifiedShuffleSplit(n_splits,test_size,random_state)
    for train_index,test_index in split.split(data,data[stratum_name]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
    
    return strat_train_set, strat_test_set

def rectify_missing_vals(data,column_name,method="median"):
    if method == "median":
        data[column_name].fillna(median,inplace=True)
    elif method == "remove rows":
        data.dropna(subset=[column_name],inplace=True)
    elif method == "remove feature":
        data.drop(column_name,axis=1)
    else:
        data[column_name].fillna(mean,inplace=True)
        
    return data
            
    
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def scale_data(train_data,method="standardization"):
    if method == "standardization":
        standardScaler = StandardScaler()
        standardScaler.fit_transform(train_data)
    else:
        minmaxScaler = MinMaxScaler()
        minmaxScaler.fit_transform(train_data)
        
    return train_data
        
def disply_cross_val_info(scores):
    print("Scores:",scores)
    print("Mean:",scores.mean())
    print("Standard Deviation:",scores.std())
    
def plot_digit(data):
    image = data.reshape(28,28)
    plt.imshow(image,cmap = mpl.cm.binary,
               interpolation = "nearest")
    plt.axis("off")
    
    
def plot_digits(instances,images_per_row = 10, **options):
    size = 28
    images_per_row = min(len(instances),images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size,size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
    

def plot_precision_recall_vs_threshold(precisions,recalls,thresholds,axis=[-20000,20000]):
    plt.plot(thresholds,precisions[:-1],"b--",label="Precision")
    plt.plot(thresholds,recalls[:-1],"g-",label="Recall")
    plt.legend(loc="center right", fontsize=16) 
    plt.xlabel("Threshold", fontsize=16)        
    plt.grid(True)                              
    plt.axis([axis[0], axis[1], 0, 1])
    
    plt.show()