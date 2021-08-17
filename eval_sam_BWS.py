# Continue training of pre-trained model
"""
This example loads the pre-trained argueBERT model and performs a similar argument mining evaluation on the BWS dataset.

Note: In this example, you must specify a argueBERT model.

author: Maike Behrendt
source: https://github.com/UKPLab/sentence-transformers
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
from SimilarArgumentMiningEvaluator import SimilarArgumentEvaluator
import logging
from datetime import datetime
import sys

sentence_dict = {}
pairs_dict = {}
no_pairs_list = []
set_number = 4
# read in sentence dictionary
with open("Datasets/BWS Argument Similarity/bws_testset_sentences.txt","r") as f:
    s = f.read()
    sentence_dict = eval(s)
# read in pairs dictionary
with open("Datasets/BWS Argument Similarity/bws_testset_pairs.txt","r") as f:
    s = f.read()
    pairs_dict = eval(s)
# read in no pairs list
with open("Datasets/BWS Argument Similarity/nopairs_testset.txt","r") as f:
    s = f.read()
    no_pairs_list = eval(s)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# specify pre-trained dataset. e.g.: bert-base-nli-mean-tokens. (default)
model_name = sys.argv[1] if len(sys.argv) > 1 else  'bert-base-nli-mean-tokens'
model_save_path = 'output/eval/BWS'

# Load a pre-trained sentence transformer model
model = SentenceTransformer(model_name)

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

test_evaluator = SimilarArgumentEvaluator(sentence_dict,pairs_dict,no_pairs_list)
test_evaluator(model,output_path=model_save_path)
