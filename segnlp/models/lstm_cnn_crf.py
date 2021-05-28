





#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F

#segnlp
from .base import PTLBase

from segnlp.layer_wrappers import Reducer
from segnlp.layer_wrappers import Encoder
from segnlp.layer_wrappers import Linker
from segnlp.layer_wrappers import Labeler


class JointPN(PTLBase):

    """
    Original Paper:
    https://arxiv.org/pdf/1612.08994.pdf
    
    more on Pointer Networks:
    https://arxiv.org/pdf/1409.0473.pdf
    https://papers.nips.cc/paper/5866-pointer-networks.pdf 

    A quick read:
    https://medium.com/@sharaf/a-paper-a-day-11-pointer-networks-59f7af1a611c

    """
    
    def __init__(self,  *args, **kwargs):   
        super().__init__(*args, **kwargs)

        self.task = self.tasks[0]

        self.char_embedder = Embedder(
                                        layer = "CharEmb", 
                                        hyperparams = self.hps.get("CharEmb", {}),
                                    )
        self.encoder = Encoder(    
                                layer = "LSTM", 
                                hyperparams = self.hps.get("LSTM", {}),
                                input_size = self.agg.output_size + self.feature_dims["doc_embs"]
                                )

        self.segmenter = Segmenter(
                                layer = "CRF",
                                hyperparams = self.hps.get("CRF", {}),
                                input_size = self.encoder.output_size,
                                output_size = self.task_dims[self.task]
                                )


    @classmethod
    def name(self):
        return "JointPN"


    def forward(self, batch):

        word_embs = self.finetune(batch["token"]["word_embs"])

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
                        self.task: preds,
                        }
                }



    def loss(self, batch, forward_output:dict):
        return = self.segmenter.loss(
                                        logits = forward_output["logits"][self.task],
                                        targets = batch["seg"][self.task],
                                        mask = batch["token"]["mask"],
                                    )

     