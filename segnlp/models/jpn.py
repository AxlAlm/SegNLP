
#pytroch
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module

#segnlp
from .base import BaseModel
from segnlp import utils
from segnlp.utils import Batch


class JointPN(BaseModel):

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

        self.agg = self.add_seg_rep(
                            layer = "Agg", 
                            hyperparamaters = self.hps.get("Agg", {}),
                            input_size = self.feature_dims["word_embs"],
                            )


        self.bow = self.add_seg_embedder(
                            layer = "SegBOW",
                            hyperparamaters = self.hps.get("seg_bow", {}),
                            )


        self.seg_pos = self.add_seg_embedder(
                                        layer = "SegPos",
                                        hyperparamaters = {},
                                        )


        self.fc1 = self.add_seg_encoder(    
                                    layer = "Linear", 
                                    hyperparamaters = self.hps.get("linear_fc", {}),
                                    input_size  =   self.agg.output_size
                                                    + self.bow.output_size
                                                    + self.seg_pos.output_size,
                                    )


        self.fc2 = self.add_seg_encoder(
                                    layer = "Linear", 
                                    hyperparamaters = self.hps.get("linear_fc", {}),
                                    input_size  =   self.agg.output_size
                                                    + self.bow.output_size
                                                    + self.seg_pos.output_size,

                                )

        self.lstm_encoder = self.add_seg_encoder(    
                                            layer = "LSTM", 
                                            hyperparamaters = self.hps.get("lstm_encoder", {}),
                                            input_size = self.fc1.output_size,
                                            )


        self.lstm_decoder = self.add_seg_encoder(    
                                            layer = "LSTM", 
                                            hyperparamaters = self.hps.get("lstm_decoder", {}),
                                            input_size = self.fc2.output_size,
                                            )


        self.pointer = self.add_linker(
                                    layer = "Pointer",
                                    hyperparamaters = self.hps.get("Pointer", {}),
                                    input_size = self.lstm_decoder.output_size,
                                    )


        self.labeler =  self.add_labeler(
                                        layer = "LinearCLF",
                                        hyperparamaters = self.hps.get("linear_clf", {}),
                                        input_size = self.lstm_encoder.output_size,
                                        output_size = self.task_dims["label"]
                                        )
                
    
    @classmethod
    def name(self):
        return "JointPN"
    

    def seg_rep(self, batch: Batch):

        seg_embs = self.agg(
                            input = batch.get("token", "embs"), 
                            lengths = batch.get("seg", "lengths"),
                            span_idxs = batch.get("seg", "span_idxs"),
                            device = batch.device
                            )

        bow = self.bow(
                        input = batch.get("token", "str"), 
                        lengths = batch.get("token", "lengths"),
                        span_idxs = batch.get("seg", "span_idxs"),
                        device = batch.device
                        )

        segpos = self.seg_pos(
                            document_paragraph_id = batch.get("seg", "document_paragraph_id"), 
                            nr_paragraphs_doc = batch.get("seg", "nr_paragraphs_doc"),
                            lengths = batch.get("seg", "lengths"),
                            device = batch.device
                            )

        seg_embs = torch.cat((seg_embs, bow, segpos), dim=-1) # 


        f1c_out = self.fc1(seg_embs)
        f2c_out = self.fc2(seg_embs)

        encoder_out, states = self.lstm_encoder(
                                        input = f1c_out,
                                        lengths = batch.get("seg", "lengths"),
                                        )


        decoder_out, _  = self.lstm_decoder(
                                        input = (f2c_out, states),
                                        lengths = batch.get("seg", "lengths"),
                                        )
        
        return {    
                "encoder_out":encoder_out, 
                "decoder_out":decoder_out
                }
        

    def seg_clf(self, batch:Batch, seg_rep_out:dict) -> dict:

        label_logits, label_preds = self.labeler(input = seg_rep_out["encoder_out"])

        link_logits, link_preds  = self.pointer(
                                    input = seg_rep_out["decoder_out"],
                                    encoder_outputs = seg_rep_out["encoder_out"],
                                    mask = batch.get("seg", "mask"),
                                    device = batch.device
                                    )

        # add/save predictions
        batch.add("seg", "label", label_preds)
        batch.add("seg", "link", link_preds)

        return {
                "label_logits" : label_logits,
                "link_logits" : link_logits,
                }

                    

    def seg_loss(self, batch: Batch, seg_clf_out:dict) -> Tensor:

        label_loss = self.labeler.loss(
                                        logits = seg_clf_out["label_logits"],
                                        targets = batch.get("seg", "label")
                                    )

        link_loss = self.pointer.loss(
                                        logits = seg_clf_out["link_logits"],
                                        targets = batch.get("seg", "link")
                                    )
        

        tw = self.hps["general"]["task_weight"]
        loss = ((1 - tw) * link_loss) + ((1 - tw) * label_loss)

        return loss