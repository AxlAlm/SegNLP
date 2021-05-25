#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F

#segnlp
from .base import PTLBase
import segnlp.layers.layer_wrappers as lw


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

        self.agg = lw.ReductionLayer(
                                    layer = "Agg", 
                                    hyperparams = self.hps.get("Agg", {}),
                                    input_size = (self.feature_dims["word_embs"] * 3) + self.feature_dims["doc_embs"]
                                    )

        self.encoder = lw.EncodingLayer(    
                                        layer = "LLSTM", 
                                        hyperparams = self.hps.get("LLSTM", {}),
                                        input_size = self.feature_dims["word_embs"] * 3
                                        )

        self.pointer = lw.LinkLayer(
                                    layer = "Pointer",
                                    task = "link",
                                    hyperparams = self.hps.get("Pointer", {}),
                                    input_size = self.encoder.output_size,
                                    output_size = self.task_dims["link"]
                                    )

        self.label_clf =  lw.CLFlayer(
                                    layer = "Linear",
                                    task = "label",
                                    hyperparams = self.hps.get("SimpleCLF", {}),
                                    input_size = self.encoder.output_size,
                                    output_size = self.task_dims["label"]
                                    )


    @classmethod
    def name(self):
        return "JointPN"


    def forward(self, batch):

        unit_embs = self.agg(
                            input = batch["token"]["word_embs"], 
                            lengths = batch["unit"]["lengths"],
                            span_idxs = batch["unit"]["span_idxs"], 
                            )

        encoder_out = self.encoder(
                                    input = unit_embs,
                                    lengths = batch["unit"]["lengths"],
                                    )
    
        label_loss, label_preds =  self.label_clf(
                                                    input = encoder_out[0],
                                                    )

        link_loss, link_preds = self.pointer(
                                        input = encoder_out,
                                        seg_data = seg_output,
                                        batch = batch
                                        )

                                    
        total_loss = ((1-self.TASK_WEIGHT) * link_loss) + ((1-self.TASK_WEIGHT) * label_loss)


        return total_loss, {"preds":{
                                        "label":label_preds,
                                        "link": link_preds
                                    }
                            }