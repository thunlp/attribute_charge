# Few-Shot Charge Prediction with Discriminative Legal Attributes
Source code and datasets of COLING2018 paper: "Few-Shot Charge Prediction with Discriminative Legal Attributes". [(pdf)](http://thunlp.org/~tcc/publications/coling2018_attribute.pdf)
## Dataset
Please download the dataset [here](https://thunlp.oss-cn-qingdao.aliyuncs.com/attribute_charge.zip), unzip it and you will get three folders: "data", "data_20w", "data_38w". Then put the folder "data" under this directory. It contains following files:

* words.vec: Pre-trained word embeddings, each line contains a word and its embedding. 
* attributes: The legal attributes for each charge.
* train: data for training from small dataset.
* test: data for test from small dataset.
* valid: data for validation from small dataset.

If you want to train and test on middle dataset, please copy the files in "data_20w" folder to "data" folder.
If you want to train and test on large dataset, please copy the files in "data_38w" folder to "data" folder.
## Run
Run the following command for training our model:

    cd code/
    python train.py

## Dependencies
* Tensorflow == 0.12
* Scipy == 0.18.1
* Numpy == 1.11.2
* Python == 2.7

## Log
After start training,  a new folder "log" will be created.There are 4 directories in it:

* /evaluation_charge_log/: stores model's performance of charge prediction on test data during training.
* /evaluation_attr_log/: stores model's performance of attribute prediction on test data during training.
* /validation_charge_log/: stores model's performance of charge prediction on validation data during training.
* /validation_attr_log/: stores model's performance of attribute prediction on validation data during training.

## Cite
If you use the code, please cite this paper:
  
Zikun Hu, Xiang Li, Cunchao Tu, Zhiyuan Liu, Maosong Sun. Few-Shot Charge Prediction with Discriminative Legal Attributes. The 27th Iinternational Conference on Computational Liguisitics (COLING 2018).

For more related works, please refer to my [homepage](http://thunlp.org/~tcc/).
