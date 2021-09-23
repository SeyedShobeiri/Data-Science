import os
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
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