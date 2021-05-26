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
                            input_size = (self.feature_dims["word_embs"] * 3) + self.feature_dims["doc_embs"]
                            )

        self.encoder = Encoder(    
                                layer = "LLSTM", 
                                hyperparams = self.hps.get("LLSTM", {}),
                                input_size = self.feature_dims["word_embs"] * 3
                                )

        self.pointer = Linker(
                                layer = "Pointer",
                                hyperparams = self.hps.get("Pointer", {}),
                                input_size = self.encoder.output_size,
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


    def forward(self, batch):

        seg_embs = self.agg(
                            input = batch["token"]["word_embs"], 
                            lengths = batch["seg"]["lengths"],
                            span_idxs = batch["seg"]["span_idxs"], 
                            )

        encoder_out = self.encoder(
                                    input = seg_embs,
                                    lengths = batch["seg"]["lengths"],
                                    )
    
        label_loss, label_preds =  self.labeler(
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