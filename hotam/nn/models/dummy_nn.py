
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

        self.task2labels = task2labels
        self.output_layers = nn.ModuleDict()
        for task, labels in task2labels.items():
            print(len(labels), labels)
            self.output_layers[task] = nn.Linear(feature2dim["dummy"], len(labels))

        self.loss = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)

  

    @classmethod
    def name(self):
        return "DummyNN"


    def forward(self, batch):
        
        print("HELLO", batch["deprel"].shape, batch["dephead"].shape)
        print(batch["deprel"])
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
        
            dense_out = output_layer(embs)

            #magic
            targets = batch[task]
            print("MAX", torch.max(targets))

            if level == "word":
                targets = targets.view(-1)
                preds = torch.flatten(dense_out, end_dim=-2)

            if level == "doc":
                targets = targets.view(-1)
                preds = torch.flatten(dense_out, end_dim=-2)
            
            print(preds.shape, targets.shape)
            loss = self.loss(preds, targets)

            #magic
            tasks_preds[task] = targets
            tasks_loss[task] = loss
            #tasks_probs[task] =  probs

        return {    
                    "loss":tasks_loss, 
                    "preds":tasks_preds,
                    "probs": {}
                }
