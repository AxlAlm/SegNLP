





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


    def __init__(self,  *args, **kwargs) -> None:   
        super().__init__(*args, **kwargs)

        self.finetuner = self.add_encoder(
                                    layer = "LinearRP", 
                                    hyperparams = {},
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
    def name(self) -> str:
        return "LSTM_CNN_CRF"


    def token_rep(self, batch: Batch) -> dict:

        #fine tune embedding via a linear layer
        word_embs = self.finetuner(batch.get("token", "word_embs"))

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
                                    input=output.stuff["lstm_out"],
                                    mask=batch.get("token", "mask"),
                                    )
        return [
                {
                "task": self.seg_task,
                "logits": logits,
                "preds": preds,
                }
                ]


    def token_loss(self, batch: Batch, token_clf_out: dict) -> Tensor:
        return self.segmenter.loss(
                                        logits = output.logits[self.seg_task],
                                        targets = batch.get("token", self.seg_task),
                                        mask = batch.get("token", "mask"),
                                    )