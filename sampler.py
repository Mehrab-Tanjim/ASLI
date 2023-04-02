import numpy as np
from multiprocessing import Process, Queue


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def random_neq_act(l, r, s):
    t = np.random.randint(l, r)
    while t == s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, categorynum, batch_size, maxlen, args, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while  user not in user_train or len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        item_seq = np.zeros([maxlen], dtype=np.int32)
        cat_seq = np.zeros([maxlen], dtype=np.int32)
        act_seq = np.zeros([maxlen], dtype=np.int32)

        
        pos_item = np.zeros([maxlen], dtype=np.int32)
        neg_item = np.zeros([maxlen], dtype=np.int32)
        
        pos_cat = np.zeros([maxlen], dtype=np.int32)
        neg_cat = np.zeros([maxlen], dtype=np.int32)
        
        pos_act = np.zeros([maxlen], dtype=np.int32)
        neg_act = np.zeros([maxlen], dtype=np.int32)

        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set()
        for i in user_train[user]:
            ts.add(i[0])
        cs = set()
        for i in user_train[user]:
            cs.add(i[3])

        for i in reversed(user_train[user][:-1]):
            item_seq[idx] = i[0]
            act_seq[idx] = i[2]
            cat_seq[idx] = i[3]

            pos_item[idx] = nxt[0]
            pos_act[idx] = nxt[2]
            pos_cat[idx] = nxt[3]

            if nxt[0] != 0: 
                neg_item[idx] = random_neq(1, itemnum + 1, ts)
                neg_act[idx] = random_neq_act(1, 4 + 1, nxt[2])
                neg_cat[idx] = random_neq(1, categorynum + 1, cs)

            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, item_seq, act_seq, cat_seq, pos_item, neg_item, pos_act, neg_act, pos_cat, neg_cat)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, categorynum, args, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      categorynum,
                                                      batch_size,
                                                      maxlen,
                                                      args,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()