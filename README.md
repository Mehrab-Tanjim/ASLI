# ASLI: Attentive Sequential model of Latent Intent

This is a TensorFlow implementation for the paper:

*[Attentive Sequential Models of Latent Intent
for Next Item Recommendation](https://dl.acm.org/doi/pdf/10.1145/3366423.3380002)*. In Proceedings of The Web Conference (WWW'20)

Please cite our paper if you use the code or dataset:

    @inproceedings{tanjim2020attentive,
        title={Attentive sequential models of latent intent for next item recommendation},
        author={Tanjim, Md Mehrab and Su, Congzhe and Benjamin, Ethan and Hu, Diane and Hong, Liangjie and McAuley, Julian},
        booktitle={Proceedings of The Web Conference 2020},
        pages={2528--2534},
        year={2020}
    }


The code is tested with TensorFlow 1.15 and Python 3.6.

## Dataset

### Tmall Dataset

This is a dataset from Alibaba e-commerce platform. The preprocessed dataset is included in the repo (`data/tmall.pickle`), where each line contains a sequence of user's past interactions (sorted by timestamp). In each sequence, we include the following tuple: (`item id`, `action+category id`, `action id`, `category id`). There are four actions: click, collect,  add-to-cart, and payment. The `action+category id` concatenates each unique combination action types along with the category id into a new id (not currently used in the model)

For downloading the original dataset:   

* Download: [Tmall](https://tianchi.aliyun.com/dataset/46)
  

## Model Training

To train our model on `Tmall` (with default hyper-parameters), simply run: 

```
python main.py 
```

If you want to experiment with changing other parameters, following are the important parameters to change:

```
'--dataset': Name of the dataset
'--train_dir': Name of the directory for experimentation
'--attention_type': Type of the attention: latent_intent, self, item, action, category, action_category (please see the paper for more details, default=latent_intent)
'--batch_size': Default=128
'--maxlen': Maximum length of the sequence (default=300)
'--hidden_units': Embedding dimension (default=200)
'--dropout_rate': Dropout in FFN (default=0.3) 
'--num_heads': Number of attention heads (default=1)
'--num_blocks': Number of attention layers (default=2) 
'--kernel_size': Kernel size for the convolution layer (default = 10)
```

## Results

On the Tmall dataset, below is an ablation study of different attention types:

| Attention Type     |  NDCG@5 |  HR@5 | 
|--------------------|--------------|-------------|
| Seq-Seq [(SASRec)](https://github.com/kang205/SASRec)    | 0.4588      | 0.5358      | 
| Seq-Item            | 0.3958       | 0.4573      | 
| Seq-Action          | 0.3928      |  0.4682      | 
| Seq-Category        | 0.5029     | 0.5791    | 
| Seq-Action+Category        |  0.4928     | 0.5687   | 
| Seq-Latent Intent   | **0.5107**   | **0.5979**  | 

As can be seen from the above table, paying attention to the proposed latent intent of a user significantly improves the recommendation for the e-commerce platform. 
