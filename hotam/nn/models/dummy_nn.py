
#basics
import numpy as np
import time

#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F

#hotam
import hotam.utils as u


class DummyNN(nn.Module):


    def __init__(self, hyperparamaters:dict, task2labels:dict, feature2dim:dict):
        super().__init__()
        self.OPT = hyperparamaters["optimizer"]
        self.LR = hyperparamaters["lr"]
        self.HIDDEN_DIM = hyperparamaters["hidden_dim"]
        self.NUM_LAYERS = hyperparamaters["num_layers"]
        self.WORD_EMB_DIM = feature2dim["word_embs"]

        self.task2labels = task2labels
        self.output_layers = nn.ModuleDict()
        for task, labels in task2labels.items():
            self.output_layers[task] = nn.Linear(self.WORD_EMB_DIM, len(labels))

        self.loss = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)

  

    @classmethod
    def name(self):
        return "DummyNN"


    def forward(self, batch):
        word_embs = batch["word_embs"]        

        tasks_preds = {}
        tasks_loss = {}
        tasks_probs = {}

        for task, output_layer in self.output_layers.items():

            dense_out = output_layer(word_embs)

            #magic
            preds = batch[task]

            loss = self.loss(torch.flatten(dense_out, end_dim=-2), preds.view(-1))

            #magic
            tasks_preds[task] = preds
            tasks_loss[task] = loss
            tasks_probs[task] =  probs

        return {    
                    "loss":tasks_loss, 
                    "preds":tasks_preds,
                    "probs": tasks_probs
                }
