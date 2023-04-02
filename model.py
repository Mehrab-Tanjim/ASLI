from modules import *


class Model():
    def __init__(self, usernum, itemnum, categorynum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.item_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.act_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.cat_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        
        self.pos_item = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg_item = tf.placeholder(tf.int32, shape=(None, args.maxlen))

        self.pos_cat = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg_cat = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        
        self.pos_act = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg_act = tf.placeholder(tf.int32, shape=(None, args.maxlen))

        pos_item = self.pos_item
        neg_item = self.neg_item

        pos_act = self.pos_act
        neg_act = self.neg_act
        
        pos_cat = self.pos_cat
        neg_cat = self.neg_cat

        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.item_seq, 0)), -1)

        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            self.seq, item_emb_table = embedding(self.item_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )
            self.item_embedding = self.seq
            # pos_itemitional Encoding
            t, pos_item_emb_table = embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(self.item_seq)[1]), 0), [tf.shape(self.item_seq)[0], 1]),
                vocab_size=args.maxlen,
                num_units=args.hidden_units,
                zero_pad=False,
                scale=False,
                l2_reg=args.l2_emb,
                scope="dec_pos_item",
                reuse=reuse,
                with_t=True
            )
            self.seq += t

            # Dropout
            self.seq = tf.layers.dropout(self.seq,
                                         rate=args.dropout_rate,
                                         training=tf.convert_to_tensor(self.is_training))
            self.seq *= mask
            
            self.action_embedding, action_embedding_table = embedding(self.act_seq,
                                                 vocab_size = 4 + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="action_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )
            
            self.cat_embedding, cat_embedding_table = embedding(self.cat_seq,
                                                 vocab_size = categorynum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="cat_embeddings",
                                                 with_t=True,
                                                 reuse=reuse
                                                 )

            # Build blocks
            self.action_plus_cat = self.action_embedding + self.cat_embedding

            # attention to item only, cat only, action only, action+cat, or latent intent  

            if args.attention_type == 'latent_intent':
                self.attention_type = lightconv1d(inputs = self.action_plus_cat, kernel_size = args.kernel_size, num_heads = 1, args=args)
            elif args.attention_type == 'self':
                self.attention_type = self.seq
            elif args.attention_type == 'category':
                self.attention_type = self.cat_embedding
            elif args.attention_type == 'item':
                self.attention_type = self.item_embedding
            elif args.attention_type == 'action':
                self.attention_type = self.action_embedding
            elif args.attention_type == 'action_category':
                self.attention_type = self.action_plus_cat

            self.interaction_output = feedforward(normalize(self.attention_type), num_units=[args.hidden_units, args.hidden_units],
                                        dropout_rate=args.dropout_rate, is_training=self.is_training)
            self.interaction_output *= mask
            self.interaction_output = normalize(self.interaction_output)

            
            with tf.variable_scope("item_similarity_block"):

                # Self-attention
                self.seq, self.self_attention = multihead_attention(queries=normalize(self.seq),
                                                keys=self.seq,
                                                num_units=args.hidden_units,
                                                num_heads=args.num_heads,
                                                dropout_rate=args.dropout_rate,
                                                is_training=self.is_training,
                                                causality=True,
                                                scope="self_attention")

                # Feed forward
                self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                        dropout_rate=args.dropout_rate, is_training=self.is_training)
                self.seq *= mask

            with tf.variable_scope("attention_block"):

                # Attention to attention type
                self.seq, self.attention_to_attention_type = multihead_attention(queries=normalize(self.attention_type), #TODO change here for example, self.item_embedding, self.action_embeeding, and self.category_embedding
                                                keys=self.seq,
                                                num_units=args.hidden_units,
                                                num_heads=args.num_heads,
                                                dropout_rate=args.dropout_rate,
                                                is_training=self.is_training,
                                                causality=True,
                                                scope="self_attention")

                # Feed forward
                self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units],
                                        dropout_rate=args.dropout_rate, is_training=self.is_training)
                self.seq *= mask

            self.seq = normalize(self.seq)

        pos_cat = tf.reshape(pos_cat, [tf.shape(self.item_seq)[0] * args.maxlen])
        neg_cat = tf.reshape(neg_cat, [tf.shape(self.item_seq)[0] * args.maxlen])

        pos_act = tf.reshape(pos_act, [tf.shape(self.item_seq)[0] * args.maxlen])
        neg_act = tf.reshape(neg_act, [tf.shape(self.item_seq)[0] * args.maxlen])

        pos_item = tf.reshape(pos_item, [tf.shape(self.item_seq)[0] * args.maxlen])
        neg_item = tf.reshape(neg_item, [tf.shape(self.item_seq)[0] * args.maxlen])

        pos_cat_emb = tf.nn.embedding_lookup(cat_embedding_table, pos_cat)
        neg_cat_emb = tf.nn.embedding_lookup(cat_embedding_table, neg_cat)

        pos_act_emb = tf.nn.embedding_lookup(action_embedding_table, pos_act)
        neg_act_emb = tf.nn.embedding_lookup(action_embedding_table, neg_act)

        
        pos_interaction_emb = pos_act_emb + pos_cat_emb
        neg_interaction_emb = neg_act_emb + neg_cat_emb

        pos_item_emb = tf.nn.embedding_lookup(item_emb_table, pos_item)
        neg_item_emb = tf.nn.embedding_lookup(item_emb_table, neg_item)

        seq_emb = tf.reshape(self.seq, [tf.shape(self.item_seq)[0] * args.maxlen, args.hidden_units])
        interaction_output_emb = tf.reshape(self.interaction_output, [tf.shape(self.item_seq)[0] * args.maxlen, args.hidden_units])


        # prediction layer
        self.pos_interaction_logits = tf.reduce_sum(pos_interaction_emb * interaction_output_emb, -1)
        self.neg_interaction_logits = tf.reduce_sum(neg_interaction_emb * interaction_output_emb, -1)

        self.pos_item_logits = tf.reduce_sum(pos_item_emb * seq_emb, -1)
        self.neg_item_logits = tf.reduce_sum(neg_item_emb * seq_emb, -1)


        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos_item, 0)), [tf.shape(self.item_seq)[0] * args.maxlen])

        self.interaction_loss = tf.reduce_sum(
                - tf.log(tf.sigmoid(self.pos_interaction_logits) + 1e-24) * istarget -
                tf.log(1 - tf.sigmoid(self.neg_interaction_logits) + 1e-24) * istarget
            ) / tf.reduce_sum(istarget)

        self.ranking_loss = tf.reduce_sum(
            - tf.log(tf.sigmoid(self.pos_item_logits) + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(self.neg_item_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)

        self.loss = self.interaction_loss + self.ranking_loss

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss += sum(reg_losses)

        self.test_item = tf.placeholder(tf.int32, shape=(None))
        test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb))
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.item_seq)[0], args.maxlen, tf.shape(self.test_item)[0]])
        self.test_logits = self.test_logits[:, -1, :]

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_item_logits - self.neg_item_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, item_seq, act_seq, cat_seq, item_idx):
        return sess.run(self.test_logits,
                        {self.u: u, self.item_seq: item_seq, self.act_seq: act_seq, self.cat_seq: cat_seq, self.test_item: item_idx, self.is_training: False})

    def getweight(self, sess, u, item_seq, cat_seq, act_seq, item_idx):
        return sess.run([self.test_logits, self.self_attention, self.attention_to_attention_type], 
                        {self.u: u, self.item_seq: item_seq, self.cat_seq: cat_seq, self.act_seq: act_seq, self.test_item: item_idx, self.is_training: False})