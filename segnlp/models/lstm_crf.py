





#pytroch
from torch import Tensor

#segnlp
from .base import BaseModel
from segnlp import utils
from segnlp.utils import Batch


class LSTM_CRF(BaseModel):

    """

    Paper using the model for Argument Mining:
    https://aclanthology.org/W19-4501.pdf

    Model paper:
    https://alanakbik.github.io/papers/coling2018.pdf


    """

    def __init__(self,  *args, **kwargs) -> None:   
        super().__init__(*args, **kwargs)

        self.finetuner = self.add_token_encoder(
                                    layer = "Linear", 
                                    hyperparamaters = self.hps.get("LinearFineTuner", {}),
                                    input_size = self.feature_dims["word_embs"],
                                )

        self.lstm = self.add_token_encoder(    
                                layer = "LSTM", 
                                hyperparamaters = self.hps.get("LSTM", {}),
                                input_size = self.finetuner.output_size,
                                )

        self.crf = self.add_segmenter(
                                layer = "CRF",
                                hyperparamaters = self.hps.get("CRF", {}),
                                input_size = self.lstm.output_size,
                                output_size = self.task_dims[self.seg_task],
                                )

        self.binary_token_dropout = self.add_token_dropout(
                                                layer = "BinaryTokenDropout",
                                                hyperparamaters = self.hps.get("token_dropout", {})
                                                )


        self.paramater_dropout = self.add_token_dropout(
                                                layer = "ParamaterDropout",
                                                hyperparamaters = self.hps.get("paramater_dropout", {})
                                                )


    def token_rep(self, batch: Batch) -> dict:

        embs = batch.get("token", "embs")

        # drop random tokens
        embs = self.binary_token_dropout(embs)

        # drop random paramaters
        embs = self.paramater_dropout(embs)

        #fine tune embedding via a linear layer
        word_embs = self.finetuner(batch.get("token", "embs"))

        # lstm encoder
        lstm_out, _ = self.lstm(
                                    input = word_embs, 
                                    lengths = batch.get("token", "lengths")
                                )

        # drop random tokens
        lstm_out = self.binary_token_dropout(lstm_out)

        # drop random paramaters
        lstm_out = self.paramater_dropout(lstm_out)

        return {
                "lstm_out": lstm_out
                }


    def token_clf(self, batch: Batch, token_rep_out:dict) -> dict:
        logits, preds  = self.crf(
                                input=token_rep_out["lstm_out"],
                                mask=batch.get("token", "mask"),
                                )

        #add/save predictions 
        batch.add("token", self.seg_task, preds)

        return {"logits" : logits}


    def token_loss(self, batch: Batch, token_clf_out: dict) -> Tensor:
        return self.crf.loss(
                                        logits = token_clf_out["logits"],
                                        targets = batch.get("token", self.seg_task),
                                        mask = batch.get("token", "mask"),
                                    )