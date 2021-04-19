
#basics
import numpy as np
import time

#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F

#segnlp
import segnlp.utils as u


class DummyNN(nn.Module):


    def __init__(self, hyperparamaters:dict, task_dims:dict, feature_dims:dict):
        super().__init__()
        self.OPT = hyperparamaters["optimizer"]
        self.LR = hyperparamaters["lr"]
        self.HIDDEN_DIM = hyperparamaters["hidden_dim"]
        self.NUM_LAYERS = hyperparamaters["num_layers"]

        self.output_layers = nn.ModuleDict()
        for task, output_dim in task_dims.items():

            if task == "relation":
                output_dim = 100

            self.output_layers[task] = nn.Linear(feature2dim["dummy"], output_dim)

        self.loss = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)

  

    @classmethod
    def name(self):
        return "DummyNN"


    def forward(self, batch):
        
        level = None
        if "word_embs" in batch:
            embs = batch["word_embs"] 
            level = "word"      
        else:
            embs = batch["doc_embs"]        
            level = "doc"

        tasks_preds = {}
        tasks_loss = {}
        tasks_probs = {}

        for task, output_layer in self.output_layers.items():
        
            logits = output_layer(embs)

            #magic
            targets = batch[task]

            #if level == "word":
            targets = targets.view(-1)
            logits = torch.flatten(logits, end_dim=-2)
            loss = self.loss(logits, targets)

            #magic
            tasks_preds[task] = batch[task]
            tasks_loss[task] = loss
            #tasks_probs[task] =  probs

        return {    
                    "loss":tasks_loss, 
                    "preds":tasks_preds,
                    "probs": {}
                }
