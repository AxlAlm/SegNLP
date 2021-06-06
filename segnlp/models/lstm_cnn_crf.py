





#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F

#segnlp
from .base import PTLBase
from segnlp.layer_wrappers import Embedder
from segnlp.layer_wrappers import Encoder
from segnlp.layer_wrappers import Segmenter
from segnlp.layer_wrappers import Reprojecter


from segnlp import utils


class LSTM_CNN_CRF(PTLBase):

    """
    BiLSTM CNN CRF network
    
    ARGUMENT MINING PAPER
    paper1:
    https://arxiv.org/pdf/1704.06104.pdf

    ORIGINAL PAPER INTRODUCING NETWORK
    paper2:
    https://www.aclweb.org/anthology/P16-1101.pdf

    """

    
    def __init__(self,  *args, **kwargs):   
        super().__init__(*args, **kwargs)

        self.task = self.tasks[0]

        self.finetuner = Reprojecter(
                                    layer = "LinearRP", 
                                    hyperparams = self.hps.get("LinearRP", {}),
                                    input_size = self.feature_dims["word_embs"]
                                )

        self.char_embedder = Embedder(
                                        layer = "CharEmb", 
                                        hyperparams = self.hps.get("CharEmb", {}),
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
                                #labels = self.task_labels[self.task],
                                )


    @classmethod
    def name(self):
        return "LSTM_CNN_CRF"

    #@utils.timer
    def forward(self, batch):

        word_embs = self.finetuner(batch["token"]["word_embs"])
        char_embs = self.char_embedder(batch["token"]["chars"])
        cat_emb = torch.cat((word_embs, char_embs), dim=-1)
    
        lstm_out, _ = self.encoder(
                                    input = cat_emb, 
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