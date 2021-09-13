

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

        self.finetuner = self.add_encoder(
                                    layer = "Linear", 
                                    hyperparams = {},
                                    input_size = self.feature_dims["word_embs"],
                                    module = "token_module"
                                )

        self.char_embedder = self.add_token_embedder(
                                                layer = "CharEmb", 
                                                hyperparams = self.hps.get("CharEmb", {}),
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
                                output_size = self.task_dims[self.seg_task],
                                task = self.seg_task
                                )


    @classmethod
    def name(self):
        return "LSTM_CNN_CRF"


    def token_rep(self, batch: utils.BatchInput, output: utils.BatchOutput):

        #fine-tune word embeddings by reprojecting them with a linear layer
        word_embs = self.finetuner(batch["token"]["word_embs"])

        #getting character embeddings
        char_embs = self.char_embedder(
                                        batch["token"]["str"],
                                        batch["token"]["lengths"]
                                        )
    
        # passing features to lstm (it will concatenate features)
        lstm_out, _ = self.encoder(
                                    input = (word_embs, char_embs), 
                                    lengths = batch["token"]["lengths"]
                                )

        return {
                "lstm_out":lstm_out
                }


    def token_clf(self, batch: utils.BatchInput, output: utils.BatchOutput):
        return self.segmenter(
                                    input=output.stuff["lstm_out"],
                                    mask=batch["token"]["mask"],
                                    )


    def loss(self, batch: utils.BatchInput, output: utils.BatchOutput):
        return self.segmenter.loss(
                                        logits = output.logits[self.seg_task],
                                        targets = batch["token"][self.seg_task],
                                        mask = batch["token"]["mask"],
                                    )