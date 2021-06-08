





#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F

#segnlp
from .base import PTLBase
from segnlp.layer_wrappers import Encoder
from segnlp.layer_wrappers import Segmenter
from segnlp.layer_wrappers import Reprojecter


from segnlp import utils


class LSTM_CRF(PTLBase):

    """

    """

    
    def __init__(self,  *args, **kwargs):   
        super().__init__(*args, **kwargs)

        self.task = self.tasks[0]

        self.finetuner = Reprojecter(
                                    layer = "LinearRP", 
                                    hyperparams = self.hps.get("LinearRP", {}),
                                    input_size = self.feature_dims["word_embs"]
                                )

        self.encoder = Encoder(    
                                layer = "LSTM", 
                                hyperparams = self.hps.get("LSTM", {}),
                                input_size = self.finetuner.output_size + self.char_embedder.output_size
                                )

        self.segmenter = Segmenter(
                                layer = "CRF",
                                hyperparams = self.hps.get("CRF", {}),
                                input_size = self.encoder.output_size,
                                output_size = self.task_dims[self.task],
                                task = self.task,
                                #labels = self.task_labels[self.task],
                                )


    @classmethod
    def name(self):
        return "LSTM_CNN_CRF"

    #@utils.timer
    def forward(self, batch):

        word_embs = self.finetuner(batch["token"]["word_embs"])
    
        lstm_out, _ = self.encoder(
                                    input = word_embs, 
                                    lengths = batch["token"]["lengths"]
                                )

        logits, preds = self.segmenter(
                                    input=lstm_out,
                                    mask=batch["token"]["mask"],
                                    )

        return  {
                "logits":{
                        self.task: logits,
                        },
                "preds": {
                        self.task: preds["preds"],
                        }
                }


    def loss(self, batch:dict, forward_output:dict):
        return self.segmenter.loss(
                                        logits = forward_output["logits"][self.task],
                                        targets = batch["token"][self.task],
                                        mask = batch["token"]["mask"],
                                    )