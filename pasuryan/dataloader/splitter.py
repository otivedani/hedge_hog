import random
import numpy as np

#%%
def splitter(id_array, num_subjects, split_ratio):
    """
    Parameters : 
    ------------
        id_array = 1D array-like, array of data ids
        num_subjects = sum of subjects in data (data class)
        split_ratio = test/train ratio, in float


    Return :
    --------
        (tuple) indexes of test data, indexes of train data 
    """

    #1 count all data per subjects
    countsubj = [0]*num_subjects
    for sub_i in range(num_subjects):
        for item in id_array:
            if sub_i == item:
                countsubj[sub_i] += 1

    #2 divide into ratio, use as tracker too
    countsubj_test = [int(item*split_ratio) for item in countsubj]
    
    id_arridxs_test = []
    id_arridxs_train = []
    #3 loop over data
    # for ncount in countsubj_test:
        # if (ncount > 0):
    for it_id in range(len(id_array)):
        wayahe = (random.randint(0,1) > 0.7)
        item = id_array[it_id]
        if (wayahe and (countsubj_test[item] > 0)):
            id_arridxs_test.append(it_id)
            countsubj_test[item] -= 1
        else:
            id_arridxs_train.append(it_id)
    
    return id_arridxs_test, id_arridxs_train

def splitter_random(id_array, num_subjects, split_ratio):
    """
    Parameters : 
    ------------
        id_array = 1D array-like, array of data ids
        num_subjects = sum of subjects in data (data class)
        split_ratio = test/train ratio, in float


    Return :
    --------
        (tuple) indexes of test data, indexes of train data 
    """
    # convert to numpy
    id_nparr = np.asarray(id_array)
    # get indexes
    nparr_idx = np.arange(id_nparr.size)
    # subjects arange
    id_subjs = np.arange(num_subjects)

    # pointing subjects in dataset
    _ptselector = id_nparr[None,:]==id_subjs[:,None]

    selected_test_idx = []

    for sb_j in range(num_subjects):
        ptsel_idx = np.where(_ptselector[sb_j] == True)
        seld_idx = nparr_idx[ptsel_idx]
        selected_test_idx.append(np.random.choice(seld_idx,int(seld_idx.size*split_ratio),replace=False))
        
    te_nparr_idx = np.array([var for sublist in selected_test_idx for var in sublist])
    # print np.delete(nparr_idx, te_nparr_idx)

    # id_npidxs_test = nparr_idx[te_nparr_idx]
    id_npidxs_test = te_nparr_idx.ravel()
    id_npidxs_train = np.delete(nparr_idx, te_nparr_idx.ravel())
    
    return id_npidxs_test, id_npidxs_train

# tests
# #%%
# import numpy as np

#param init
# num_subjects = 9
# num_img_per_subj = 10
# split_ratio = 0.2

# dummy_id = np.arange(num_subjects).repeat(num_img_per_subj)


# dummy_id_test, dummy_id_train = splitter(dummy_id, num_subjects, split_ratio)

# print "\n"
# print dummy_id_test
# print dummy_id_train
# print len(dummy_id_test)
# print len(dummy_id_train)
# print len(dummy_id_train)+len(dummy_id_test) == len(dummy_id)

# dummy_id_test, dummy_id_train = splitter_random(dummy_id, num_subjects, split_ratio)
# print np.all(np.sort(np.append(dummy_id_train, dummy_id_test)) == np.arange(dummy_id.size))
# #%%


#%%
