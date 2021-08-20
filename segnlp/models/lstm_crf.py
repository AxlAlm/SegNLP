





#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F

#segnlp
from .base import PTLBase
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


    def token_rep(self, batch: utils.Input, output: utils.Output):
        word_embs = self.finetuner(batch["token"]["word_embs"])
    
        lstm_out, _ = self.encoder(
                                    input = word_embs, 
                                    lengths = batch["token"]["lengths"]
                                )

        return {
                "lstm_out": lstm_out
                }


    def token_clf(self, batch: utils.Input, output: utils.Output):
        return self.segmenter(
                                    input = output.stuff["lstm_out"],
                                    mask = batch["token"]["mask"],
                                    )
        return logits, preds



    def loss(self, batch:dict, output:utils.Output):
        return self.segmenter.loss(
                                        logits = output.logits[self.task],
                                        targets = batch["token"][self.task],
                                        mask = batch["token"]["mask"],
                                    )