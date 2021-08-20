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
from segnlp.layers.embedders import BOW
from segnlp import utils


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

        self.agg = Reducer(
                            layer = "Agg", 
                            hyperparams = self.hps.get("Agg", {}),
                            input_size = self.feature_dims["word_embs"]
                            )


        self.bow = Embedder(
                            layer = "BOW",
                            #vocab = self.feature_dims["vocab"],
                            hyperparams = self.hps.get("BOW", {}),
                            )


        self.fc1 = Reprojecter(
                                    layer = "LinearRP", 
                                    hyperparams = self.hps.get("LinearRP", {}),
                                    input_size = self.bow.output_size + self.agg.output_size + self.feature_dims["doc_embs"]
                                )


        self.fc2 = Reprojecter(
                                    layer = "LinearRP", 
                                    hyperparams = self.hps.get("LinearRP", {}),
                                    input_size = self.bow.output_size + self.agg.output_size + self.feature_dims["doc_embs"]
                                )


        self.encoder = Encoder(    
                                layer = "LSTM", 
                                hyperparams = self.hps.get("LSTM", {}),
                                input_size = self.fc1.output_size
                                )


        self.decoder = Encoder(    
                                layer = "LSTM", 
                                hyperparams = self.hps.get("LSTM2", {}),
                                input_size = self.fc1.output_size
                                )


        self.pointer = Linker(
                                layer = "Pointer",
                                hyperparams = self.hps.get("Pointer", {}),
                                input_size = self.decoder.output_size,
                                output_size = self.task_dims["link"]
                                )


        self.labeler =  Labeler(
                                layer = "LinearCLF",
                                hyperparams = self.hps.get("LinearCLF", {}),
                                input_size = self.encoder.output_size,
                                output_size = self.task_dims["label"]
                                )
        
    

    @classmethod
    def name(self):
        return "JointPN"
    

    def seg_rep(self, batch, output):
        seg_embs = self.agg(
                            input = batch["token"]["word_embs"], 
                            lengths = batch["seg"]["lengths"],
                            span_idxs = batch["seg"]["span_idxs"],
                            )

        bow = self.bow(
                        word_encs = batch["token"]["words"], 
                        span_idxs = batch["seg"]["span_idxs"]
                        )

        seg_embs = torch.cat((seg_embs, bow, batch["seg"]["doc_embs"]), dim=-1)

        f1c_out = self.fc1(seg_embs)
        f2c_out = self.fc2(seg_embs)

        encoder_out, states = self.encoder(
                                        input = f1c_out,
                                        lengths = batch["seg"]["lengths"],
                                        )


        decoder_out, _  = self.decoder(
                                        input = (f2c_out, states),
                                        lengths = batch["seg"]["lengths"],
                                        )
        
        return {    
                "encoder_out":encoder_out, 
                "decoder_out":decoder_out
                }
        

    def seg_clf(self, batch:utils.Input, output:utils.Output):

        logits, preds = self.labeler(
                                    input = output.stuff["encoder_out"],
                                    )

        logits, preds = self.pointer(
                                    inputs = output.stuff["decoder_out"],
                                    encoder_outputs = output.stuff["encoder_out"],
                                    mask = batch["seg"]["mask"],
                                    )

        return [
                {  
                    "task": "label",
                    "logits": link_logits, 
                    "preds": label_preds,
                    "level": "seg",
                },
                {   
                    "task":"link",
                    "logits": link_logits,
                    "preds": link_preds,
                    "level": "seg",
                },
                {   
                    "task": "link_label",
                    "logits": link_label_logits,
                    "preds": link_label_preds,
                    "level": "seg",
                }
                ]   




    def loss(self, batch, output:dict):
        
        label_loss = self.labeler.loss(
                                        logits = output.logits["label"],
                                        targets = batch["seg"]["label"]
                                    )

        link_loss = self.pointer.loss(
                                        logits = output.logits["link"],
                                        targets = batch["seg"]["link"]
                                    )
                                        

        tw = self.hps["general"]["task_weight"]
        loss = ((1 - tw) * link_loss) + ((1 - tw) * label_loss)

        return loss