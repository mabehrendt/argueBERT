from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction
import torch
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from sentence_transformers.util import batch_to_device
import os
import csv
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
from typing import List

class SimilarArgumentEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on identifying paraphrases in the set, if they are existing.
    The metric is the cosine similarity.
    The returned score is the accuracy and the F1 score.

    The results are written in a CSV. If a CSV already exists, then values are appended.

    :param sentences: Dictionary of ID's to sentences {ID: string}
    :param pairs: Dictionary of all pairs in the set {ID1: [pairID1,pairID2..]}
    :param nopairs: list of sentences without a paraphrase in the set [str]
    :param threshold: threshold to decide whether two sentences are paraphrases (default = 0.8)
    :param batch_size: Batch size used to compute embeddings
    :param show_progress_bar: If true, prints a progress bar
    """

    def __init__(self, sentences, pairs, nopairs: List[str], threshold: float = 0.8, batch_size: int = 16, show_progress_bar: bool = False):
        self.sentences = sentences
        self.pairs = pairs
        self.nopairs = nopairs
        self.threshold = threshold
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "sam_evaluation_results.csv"
        self.csv_headers = ["epoch", "steps",
                            "cosine_acc", "cosine_f1", "threshold", "cosine_precision", "cosine_recall"]


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logging.info("Similar Argument Mining Evaluation of the model" + out_txt)

        sent_list = list(self.sentences.values()) #list of all sentences
        ids_list = list(self.sentences) #list of all ids
        
		# embedd all sentences
        embeddings = model.encode(sent_list, batch_size=self.batch_size,
                                   show_progress_bar=self.show_progress_bar, convert_to_numpy=True)

        cosine_scores = 1-cosine_distances(embeddings) # take max score   [0, 0.5, 0.2, 0.6, 0.9]
        np.fill_diagonal(cosine_scores,0)

        no_paraphrase = []
        pred_pairs = []
        file_output_data = [epoch, steps]
        num_correct = 0
        num_true_pos = 0
        num_false_pos = 0
        num_false_neg = 0
        num_all = len(self.sentences)

        for i in range(len(cosine_scores)):
            max_ind = np.argmax(cosine_scores[i])
            max_score = cosine_scores[i][max_ind]
            if max_score >= self.threshold:
                pred_pairs.append([ids_list[i],ids_list[max_ind]])
            else:
                no_paraphrase.append(ids_list[i])

        for j in range(len(pred_pairs)):
            if(pred_pairs[j][0] in self.pairs.get(pred_pairs[j][1],[]) or pred_pairs[j][1] in self.pairs.get(pred_pairs[j][0],[])):     
                num_correct += 1
                num_true_pos +=1
            else:
                num_false_pos += 1

        for s in range(len(no_paraphrase)):
            if no_paraphrase[s] in self.nopairs:
                num_correct += 1
            else:
                num_false_neg += 1
     
        print("Num correct:", num_correct)
        print("Num all:",num_all)
        print("TP:", num_true_pos)
        print("FP:", num_false_pos)
        print("FN:", num_false_neg)
        acc = num_correct / num_all
        precision = num_true_pos/(num_true_pos+num_false_pos)
        recall = num_true_pos / (num_true_pos+num_false_neg)
        f1 = 2*((precision*recall)/(precision+recall))

        logging.info("Accuracy:           {:.2f}\t(Threshold: {:.4f})".format(acc * 100, self.threshold))
        logging.info("F1:                 {:.2f}\t(Threshold: {:.4f})".format(f1 * 100, self.threshold))

