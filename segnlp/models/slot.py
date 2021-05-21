#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F

#segnlp
from segnlp.ptl import PTLBase
from segnlp.layers import SegLayer
from segnlp.layers import RepLayer


"""


If prediction_level == "token":

    finetuning layer
    char_emb_layer
    
    encodinglayer
    







"""

class SLOT(PTLBase):

    
    def __init__(self,  *args, **kwargs):   
        super().__init__(*args, **kwargs)

        self.agg = ReductionLayer(
                                    layer = "Agg", 
                                    hyperparams = self.hps.get("Agg", {}),
                                    input_size = (self.feature_dims["word_embs"] * 3) + self.feature_dims["doc_embs"]
                                    )

        self.encoder = EncodingLayer(    
                                        layer = "LLSTM", 
                                        hyperparams = self.hps.get("LLSTM", {}),
                                        input_size = self.feature_dims["word_embs"] * 3
                                        )

        self.pointer = SegLayer(
                            layer = "Pointer",
                            task = "link",
                            hyperparams = self.hps.get("Pointer", {}),
                            input_size = self.encoder.output_size,
                            output_size = self.task_dims["link"]
                            )

        self.label_clf =  SegLayer(
                                    layer = "SimpleCLF",
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