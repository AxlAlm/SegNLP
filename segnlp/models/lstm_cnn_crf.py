

#pytroch
import torch
from torch.functional import Tensor
import torch.nn as nn

#segnlp
from segnlp.seg_model import SegModel
from segnlp.utils import Batch


class LSTM_CNN_CRF(SegModel):

    """
    BiLSTM CNN CRF network
    
    ARGUMENT MINING PAPER
    paper1:
    https://arxiv.org/pdf/1704.06104.pdf
    (https://aclanthology.org/P17-1002.pdf)


    original code:
    https://github.com/UKPLab/acl2017-neural_end2end_AM

    ORIGINAL PAPER INTRODUCING NETWORK
    paper2:
    https://www.aclweb.org/anthology/P16-1101.pdf

    """

    def __init__(self,  *args, **kwargs):   
        super().__init__(*args, **kwargs)

        self.word_embs = self.add_token_embedder(
                                    layer = "PretrainedEmbs", 
                                    hyperparamaters = self.hps.get("word_embs", {}),
                                )

        self.char_embedder = self.add_token_embedder(
                                                layer = "CharWordEmb", 
                                                hyperparamaters = self.hps.get("CharEmb", {}),
                                                )

        self.lstm_encoder = self.add_token_encoder(    
                                layer = "LSTM", 
                                hyperparamaters = self.hps.get("LSTM", {}),
                                input_size = self.word_embs.output_size + self.char_embedder.output_size,
                                )

        self.dropout = self.add_token_dropout(
                                                layer = nn.Dropout,
                                                hyperparamaters = self.hps.get("dropout", {})
                                                )

        self.segmenter = self.add_segmenter(
                                layer = "CRF",
                                hyperparamaters = self.hps.get("CRF", {}),
                                input_size = self.lstm_encoder.output_size,
                                output_size = self.task_dims[self.seg_task],
                                )

        

    @classmethod
    def name(self) -> str:
        return "LSTM_CNN_CRF"


    def token_rep(self, batch: Batch):

        #fine-tune word embeddings by reprojecting them with a linear layer
        word_embs = self.word_embs(
                                    input = batch.get("token", "str"),
                                    lengths = batch.get("token", "lengths"),
                                    device = batch.device
                                    )

        #getting character embeddings
        char_embs = self.char_embedder(
                                        input = batch.get("token", "str"),
                                        lengths = batch.get("token", "lengths"),
                                        device = batch.device
                                        )
    
        # passing features to lstm (it will concatenate features)
        lstm_out, _ = self.lstm_encoder(
                                    input = (word_embs, char_embs), 
                                    lengths = batch.get("token", "lengths")
                                )

        # dropout on lstm output
        lstm_out = self.dropout(lstm_out)

        return {
                "lstm_out":lstm_out
                }


    def token_clf(self, batch: Batch, token_rep_out: dict) -> dict:
        logits, preds  = self.segmenter(
                                    input = token_rep_out["lstm_out"],
                                    mask = batch.get("token", "mask"),
                                    )

        # add/save prediction
        batch.add("token", self.seg_task, preds)
        return {"logits" : logits}


    def token_loss(self, batch: Batch, token_clf_out:dict) -> Tensor:
        return self.segmenter.loss(
                                        logits = token_clf_out["logits"],
                                        targets = batch.get("token", self.seg_task),
                                        mask = batch.get("token", "mask"),
                                    )