Code of Contextualized Knowledge Graph Embedding for Explainable Talent Training Course Recommendation.
Enjoy the code.

**************************** Requirement ****************************
#requirement python 3.6,pytorch 1.3,cuda 10.0,cudnn 7.6

******************************* USAGE *******************************
data_read.py ----- The code of dataloader.
transformers1.py  ----- The code of transformer module.
CKGE.py  ----- The code of model
run1.py  ----- The main code of the algorithm, including evaluation methods. It takes training data/label and parameters as input.

--the data file should contain:
#'e2c_vocab':  dataset vocab
#'e2c_train': train data
#'e2c_valid': valid data
#'e2c_test': test data
#'e2c_kg': auxiliary kg data
#'short_dis': entity short distance data
#'path': train pair path data


--the parameters
In the main
#'lr': the learning rate
#'embed_size': the embedding size of entity
#'nhead': the parameter of transformer head
#'nhid': the parameter of transformer hidden dimension
#'nlayers': the parameter of transformer layers

--demo:
i.e.
python run1.py

***************************** REFERENCE *****************************
If you use this code in scientific work, please cite:
Yang Yang, Chu-Bing Zhang, Xin Song, Zhen Dong, Heng-Shu Zhu, Wen-Jie Li, Jian yang. Contextualized Knowledge Graph Embedding for Explainable Talent Training Course Recommendation.
*********************************************************************