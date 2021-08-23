





#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F

#segnlp
from .base import PTLBase
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

        self.finetuner = self.add_encoder(
                                    layer = "LinearRP", 
                                    hyperparams = self.hps.get("LinearRP", {}),
                                    input_size = self.feature_dims["word_embs"],
                                    module = "token_module"
                                )

        self.char_embedder = self.add_embedder(
                                        layer = "CharEmb", 
                                        hyperparams = self.hps.get("CharEmb", {}),
                                        module = "token_module"
                                    )

        self.encoder = self.add_encoder(    
                                layer = "LSTM", 
                                hyperparams = self.hps.get("LSTM", {}),
                                input_size = self.finetuner.output_size + self.char_embedder.output_size,
                                module = "token_module"
                                )

        self.segmenter = self.add_segmenter(
                                layer = "CRF",
                                hyperparams = self.hps.get("CRF", {}),
                                input_size = self.encoder.output_size,
                                output_size = self.task_dims[self.task],
                                task = self.seg_task
                                )


    @classmethod
    def name(self):
        return "LSTM_CNN_CRF"


    def token_rep(self, batch: utils.Input, output: utils.Output):
        word_embs = self.finetuner(batch["token"]["word_embs"])
        char_embs = self.char_embedder(batch["token"]["chars"])
        cat_emb = torch.cat((word_embs, char_embs), dim=-1)
    
        lstm_out, _ = self.encoder(
                                    input = cat_emb, 
                                    lengths = batch["token"]["lengths"]
                                )

        return {
                "lstm_out":lstm_out
                }

    def token_clf(self, batch: utils.Input, output: utils.Output):
        return self.segmenter(
                                    input=output.stuff["lstm_out"],
                                    mask=batch["token"]["mask"],
                                    )


    def loss(self, batch: utils.Input, output: utils.Output):
        return self.segmenter.loss(
                                        logits = output.logits[self.task],
                                        targets = batch["token"][self.task],
                                        mask = batch["token"]["mask"],
                                    )