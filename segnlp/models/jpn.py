#pytroch
from segnlp.layer_wrappers.layer_wrappers import Embedder, Reprojecter
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

        self.pointer = Linker(
                                layer = "Pointer",
                                hyperparams = self.hps.get("Pointer", {}),
                                input_size = self.fc2.output_size,
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
        
        return {    
                "encoder_out":encoder_out, 
                "states": states,
                "f1c_out":f1c_out, 
                "f2c_out":f2c_out
                }
        

    def label_clf(self, batch, output):
        logits, preds = self.labeler(
                                    input = output.stuff["encoder_out"],
                                    )
        return logits, preds


    def link_clf(self, batch, output):
        logits, preds = self.pointer(
                                    inputs = output.stuff["f2c_out"],
                                    encoder_outputs = output.stuff["encoder_out"],
                                    mask = batch["seg"]["mask"],
                                    states = output.stuff["states"],
                                    )
        return logits, preds



    # def forward(self, batch, output):

    #     seg_embs = self.agg(
    #                         input = batch["token"]["word_embs"], 
    #                         lengths = batch["seg"]["lengths"],
    #                         span_idxs = batch["seg"]["span_idxs"],
    #                         )

    #     bow = self.bow(
    #                     word_encs = batch["token"]["words"], 
    #                     span_idxs = batch["seg"]["span_idxs"]
    #                     )

    #     seg_embs = torch.cat((seg_embs, bow, batch["seg"]["doc_embs"]), dim=-1)

        
    #     f1c_out = self.fc1(seg_embs)
    #     encoder_out, states = self.encoder(
    #                                     input = f1c_out,
    #                                     lengths = batch["seg"]["lengths"],
    #                                     )
    
    #     output.add(self.labeler(
    #                     input = encoder_out,
    #                 ))


    #     f2c_out = self.fc2(seg_embs)
    #     output.add(self.pointer(
    #                     inputs = f2c_out,
    #                     encoder_outputs = encoder_out,
    #                     mask = batch["seg"]["mask"],
    #                     states = states,
    #                     ))



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