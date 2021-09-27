





#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module
from torch import Tensor

#segnlp
from .base import BaseModel
from segnlp import utils
from segnlp.utils import Batch


class LSTM_CRF(BaseModel):

    """
    https://aclanthology.org/W19-4501.pdf
    
    """

    def __init__(self,  *args, **kwargs) -> None:   
        super().__init__(*args, **kwargs)

        self.finetuner = self.add_token_encoder(
                                    layer = "Linear", 
                                    hyperparamaters = {},
                                    input_size = self.feature_dims["word_embs"],
                                )

        self.encoder = self.add_token_encoder(    
                                layer = "LSTM", 
                                hyperparamaters = self.hps.get("LSTM", {}),
                                input_size = self.finetuner.output_size,
                                )

        self.segmenter = self.add_segmenter(
                                layer = "CRF",
                                hyperparamaters = self.hps.get("CRF", {}),
                                input_size = self.encoder.output_size,
                                output_size = self.task_dims[self.seg_task],
                                )


    def token_rep(self, batch: Batch) -> dict:

        #fine tune embedding via a linear layer
        word_embs = self.finetuner(batch.get("token", "embs"))

        # lstm encoder
        lstm_out, _ = self.encoder(
                                    input = word_embs, 
                                    lengths = batch.get("token", "lengths")
                                )

        return {
                "lstm_out": lstm_out
                }


    def token_clf(self, batch: Batch, token_rep_out:dict) -> dict:
        logits, preds  = self.segmenter(
                                    input=token_rep_out["lstm_out"],
                                    mask=batch.get("token", "mask"),
                                    )

        #add/save predictions 
        batch.add("token", self.seg_task, preds)

        return {"logits" : logits}


    def token_loss(self, batch: Batch, token_clf_out: dict) -> Tensor:
        return self.segmenter.loss(
                                        logits = token_clf_out["logits"],
                                        targets = batch.get("token", self.seg_task),
                                        mask = batch.get("token", "mask"),
                                    )