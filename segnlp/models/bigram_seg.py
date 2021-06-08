
#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F

#segnlp
from .base import PTLBase
from segnlp.layer_wrappers import Segmenter
from segnlp.layer_wrappers import Encoder

class BigramSeg(PTLBase):


    def __init__(self,  *args, **kwargs):   
        super().__init__(*args, **kwargs)

        self.word_lstm = Encoder(    
                                layer = "LSTM", 
                                hyperparams = self.hps.get("LSTM", {}),
                                input_size = self.feature_dims["word_embs"]
                                )
    
        self.task = self.tasks[0]
        self.segmenter = Segmenter(
                                layer = "BigramSeg",
                                hyperparams = self.hps.get("BigramSeg", {}),
                                input_size = self.word_lstm.output_size,
                                output_size = self.task_dims[self.task],
                                decode = False
                                )

    @classmethod
    def name(self):
        return "BigramSeg"


    def forward(self, batch):



        logits, preds, _  = self.segmenter(
                                        input=batch["token"]["word_embs"],
                                        mask=batch["token"]["mask"],
                                        )

        return  {
                "logits":{
                        self.task: logits,
                        },
                "preds": {
                        self.task: preds,
                        }
                }
