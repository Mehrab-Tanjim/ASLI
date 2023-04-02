import os
import time
import argparse
import tensorflow as tf
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tmall')
    parser.add_argument('--attention_type', default='latent_intent', help=['latent_intent', 'self', 'item', 'action', 'category', 'action_category'])
    parser.add_argument('--train_dir', default='default')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=300, type=int)
    parser.add_argument('--hidden_units', default=200, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.3, type=float) 
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--kernel_size', default=10, type=int) 
    

    try:
        #if running from command line
        args = parser.parse_args()
    except:
        #if running in ides
        args = parser.parse_known_args()[0] 


    args = parser.parse_args()

    args.train_dir = f'{args.train_dir}_{args.attention_type}'

    result_path = os.path.join('results', os.path.join(args.dataset, args.train_dir))
    os.makedirs(result_path, exist_ok=True)

    model_path = os.path.join(result_path, 'model.ckpt')

    with open(os.path.join(result_path, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()

    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum, actioncatnum, categorynum, categorymap] = dataset

    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))
    

    f = open(os.path.join(result_path, 'log.txt'), 'w')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    sampler = WarpSampler(user_train, usernum, itemnum, categorynum, args, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    model = Model(usernum, itemnum, categorynum, args)
    sess.run(tf.initialize_all_variables())

    T = 0.0
    t0 = time.time()
    
    saver = tf.train.Saver()
    best_ndcg = 0 
    stop_count = 0

    for epoch in range(1, args.num_epochs + 1):

        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            
            user, item_seq, act_seq, cat_seq, pos_item, neg_item, pos_act, neg_act, pos_cat, neg_cat = sampler.next_batch()
            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: u, model.item_seq: item_seq, model.act_seq: act_seq, model.cat_seq:cat_seq, \
                                    model.pos_item: pos_item, model.neg_item: neg_item, \
                                    model.pos_act: pos_act, model.neg_act: neg_act,\
                                    model.pos_cat: pos_cat, model.neg_cat: neg_cat, model.is_training: True})

        

        if  epoch % 5 == 0:
            t1 = time.time() - t0
            T += t1
            t_test = evaluate(model, dataset, args, sess, validation=True)
            print('Evaluating MRR, NDCG, HitRate')
            print('@5', str(t_test[:3]))
            print('@10', str(t_test[3:6]))
            print('@20', str(t_test[6:]))

            t_valid = t_test

            if t_valid[0]>best_ndcg:
                best_ndcg = t_valid[0]
                save_path = saver.save(sess, model_path)
                stop_count = 1
            else:
                stop_count += 1

            if stop_count == 5: #model did not improve 5 consequetive times            
                break

            f.write(str(t_test) + '\n')
            f.flush()
            t0 = time.time()

    saver.restore(sess, model_path)
    t_test = evaluate(model, dataset, args, sess, validation=False)
    print('Evaluating MRR, NDCG, HitRate')
    print('@5', str(t_test[:3]))
    print('@10', str(t_test[3:6]))
    print('@20', str(t_test[6:]))

    f.write('Final MRR, NDCG, HitRate'+ '\n')
    f.write('@5:' + str(t_test[:3]) + '\n')
    f.write('@10:' + str(t_test[3:6]) + '\n')
    f.write('@20:'+ str(t_test[6:]) + '\n')
    f.flush()

    f.close()
    sampler.close()
    print("Done")
