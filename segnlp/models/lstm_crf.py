

#pytroch
from torch import Tensor
import torch
from torch.nn.utils.rnn import pad_sequence


#segnlp
from segnlp.seg_model import SegModel
from segnlp.data import Batch
from segnlp import utils



class LSTM_CRF(SegModel):

    """

    Paper using the model for Argument Mining:
    https://aclanthology.org/W19-4501.pdf

    Model paper:
    https://alanakbik.github.io/papers/coling2018.pdf


    """

    def __init__(self,  *args, **kwargs) -> None:   
        super().__init__(*args, **kwargs)

        self.word_embs = self.add_token_embedder(
                                    layer = "FlairEmbeddings", 
                                    hyperparamaters = self.hps.get("flair_embeddings", {}),
                                )

        self.finetuner = self.add_token_encoder(
                                    layer = "Linear", 
                                    hyperparamaters = self.hps.get("LinearFineTuner", {}),
                                    input_size = self.word_embs.output_size,
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

        # embedd the tokens, returns a 3D tensor (n_sentence, n_toks, emb_size)
        embs = self.word_embs(
                            input=batch.get("sentence", "str", flat = True)
                            )

        ###  Three following small blocks are to convert (n_sentence, n_toks, embs_size)
        ### to (batch_size, n_toks, embs_size). batch_size and n_sentences
    
        # as embs is based on sentences and is padded to the longest sentence in the dataset
        # we simple remove the dims on axis 1 which are above the lenght of the longest sentence
        # in the batch
        max_sent_len = max(batch.get("sentence", "tok_length", flat = True))
        embs = embs[:, :max_sent_len, :]

        # we then create a mask for all sentences and then use the mask to select tokens from 
        # the 2D version of embs. 
        sent_tok_mask = utils.create_mask(batch.get("sentence", "tok_length", flat = True)).view(-1)
        flat_embs = embs.reshape(embs.size(0)*embs.size(1), embs.size(2))
        flat_embs = flat_embs[sent_tok_mask]
        
        # We then split the flat embs into size of our batch
        embs = pad_sequence(torch.split(flat_embs,
                                        utils.ensure_list(batch.get("token", "length"))
                                        ),
                            batch_first = True,
                            )
        
        # drop random tokens
        embs = self.binary_token_dropout(embs)

        # drop random paramaters
        embs = self.paramater_dropout(embs)


        #fine tune embedding via a linear layer
        embs = self.finetuner(embs)

        # lstm encoder
        lstm_out, _ = self.lstm(
                                    input = embs, 
                                    lengths = batch.get("token", "length")
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