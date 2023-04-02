import copy
import numpy as np
from tqdm import tqdm
import pickle

def data_partition(fname):
    
    pickle_in = open('data/%s.pickle'%fname,"rb")
    dataset = pickle.load(pickle_in)

    [user_train, user_valid, user_test, usernum, itemnum, actioncatnum, categorynum, categorymap] = dataset

    for u in user_train:
        for i in user_valid[u]:
            user_train[u].append(i)
        for i in user_test[u]:
            user_train[u].append(i)

    user_valid = {}
    user_test = {}

    for u in user_train:
        
        user_test[u] =  user_train[u][-1]
        del user_train[u][-1]
        user_valid[u] =  user_train[u][-1]
        del user_train[u][-1]
        
    
    return [user_train, user_valid, user_test, usernum, itemnum, actioncatnum, categorynum, categorymap]


def evaluate(model, dataset, args, sess, validation):
    [user_train, user_valid, user_test, usernum, itemnum, actioncatnum, categorynum, categorymap] = copy.deepcopy(dataset)
    
    MRR = 0.0
    MRR_10 =0.0
    MRR_20 =0.0

    HT = 0.0
    HT_10 = 0.0
    HT_20 = 0.0

    NDCG = 0.0    
    NDCG_10 = 0.0
    NDCG_20 = 0.0
        
    valid_user = 0.0


    for u in tqdm(user_train):

        valid_user += 1    
        
        item_seq = np.zeros([args.maxlen], dtype=np.int32)        
        act_seq = np.zeros([args.maxlen], dtype=np.int32)    
        cat_seq = np.zeros([args.maxlen], dtype=np.int32)

        idx = args.maxlen - 1

        if not validation:
            item_idx = [user_test[u][0]]
        else:
            item_idx = [user_valid[u][0]]

        if not validation:
            item_seq[idx] = user_valid[u][0]
            act_seq[idx] = user_valid[u][2]
            cat_seq[idx] = user_valid[u][3]        
            idx -= 1
        
        for i in reversed(user_train[u]):
            item_seq[idx] = i[0]
            act_seq[idx] = i[2]
            cat_seq[idx] = i[3]
            idx -= 1
            if idx == -1: break
         
        rated = set()
        for i in user_train[u]:
            rated.add(i[0])
        rated.add(0)

        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(sess, [u], [item_seq], [act_seq], [cat_seq], item_idx)
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0]

        if rank < 5:
            MRR += 1/float(rank + 1)
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
            
        if rank < 10:
            MRR_10 += 1/float(rank + 1)
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1
        
        if rank < 20:
            MRR_20 += 1/float(rank + 1)
            NDCG_20 += 1 / np.log2(rank + 2)
            HT_20 += 1
                    
        if validation and valid_user > 3000:
            break

    return MRR/ valid_user, NDCG / valid_user, HT / valid_user, MRR_10/ valid_user, NDCG_10 / valid_user, HT_10 / valid_user,  MRR_20 / valid_user, NDCG_20 / valid_user, HT_20 / valid_user
