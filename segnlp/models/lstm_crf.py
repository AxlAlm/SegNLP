





#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module

#segnlp
from .base import PTLBase
from segnlp import utils


class LSTM_CRF(PTLBase):

    """

    """
    
    def __init__(self,  *args, **kwargs):   
        super().__init__(*args, **kwargs)

        self.finetuner = self.add_encoder(
                                    layer = "LinearRP", 
                                    hyperparams = self.hps.get("LinearRP", {}),
                                    input_size = self.feature_dims["word_embs"],
                                    module = "token_module"
                                )

        self.encoder = self.add_encoder(    
                                layer = "LSTM", 
                                hyperparams = self.hps.get("LSTM", {}),
                                input_size = self.finetuner.output_size,
                                module = "token_module"
                                )

        self.segmenter = self.add_segmenter(
                                layer = "CRF",
                                hyperparams = self.hps.get("CRF", {}),
                                input_size = self.encoder.output_size,
                                output_size = self.task_dims[self.seg_task],
                                task = self.seg_task,
                                )


    @classmethod
    def name(self):
        return "LSTM_CNN_CRF"


    def token_rep(self, batch: utils.BatchInput, output: utils.BatchOutput):

        #fine tune embedding via a linear layer
        word_embs = self.finetuner(batch["token"]["word_embs"])

        # lstm encoder
        lstm_out, _ = self.encoder(
                                    input = word_embs, 
                                    lengths = batch["token"]["lengths"]
                                )

        return {
                "lstm_out": lstm_out
                }


    def token_clf(self, batch: utils.BatchInput, output: utils.BatchOutput):
        return self.segmenter(
                                    input = output.stuff["lstm_out"],
                                    mask = batch["token"]["mask"],
                                    )


    def loss(self, batch: utils.BatchInput, output: utils.BatchOutput):
        return self.segmenter.loss(
                                        logits = output.logits[self.seg_task],
                                        targets = batch["token"][self.seg_task],
                                        mask = batch["token"]["mask"],
                                    )