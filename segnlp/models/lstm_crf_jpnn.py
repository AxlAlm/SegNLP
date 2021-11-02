
#pytroch
import torch
from torch import Tensor

#segnlp
from segnlp.seg_model import SegModel
from segnlp.utils import Batch









#pytroch
from torch import Tensor

#segnlp
from segnlp.seg_model import SegModel
from segnlp.utils import Batch



class LSTM_CRF_JPNN(SegModel):

    
    def __init__(self,  *args, **kwargs):   
        super().__init__(*args, **kwargs)


        ### TOKEN MODULE
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


        # SEG MODULE
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
        return "LSTM_CRF_JointPointerNN"


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



    def seg_rep(self, batch: Batch, token_rep_out:dict):

        seg_embs = self.agg(
                            input = batch.get("token", "embs"), 
                            lengths = batch.get("seg", "lengths", pred = True),
                            span_idxs = batch.get("seg", "span_idxs", pred = True),
                            device = batch.device
                            )

        bow = self.bow(
                        input = batch.get("token", "str"), 
                        lengths = batch.get("token", "lengths", pred = True),
                        span_idxs = batch.get("seg", "span_idxs", pred = True),
                        device = batch.device
                        )

        segpos = self.seg_pos(
                            document_paragraph_id = batch.get("seg", "document_paragraph_id", pred = True), 
                            nr_paragraphs_doc = batch.get("seg", "nr_paragraphs_doc", pred = True),
                            lengths = batch.get("seg", "lengths", pred = True),
                            device = batch.device
                            )

        seg_embs = torch.cat((seg_embs, bow, segpos), dim=-1) # 

        f1c_out = self.fc1(seg_embs)
        f2c_out = self.fc2(seg_embs)

        encoder_out, states = self.lstm_encoder(
                                        input = f1c_out,
                                        lengths = batch.get("seg", "lengths", pred = True),
                                        )


        decoder_out, _  = self.lstm_decoder(
                                        input = (f2c_out, states),
                                        lengths = batch.get("seg", "lengths", pred = True),
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
                                    mask = batch.get("seg", "mask", pred = True),
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
                                        targets = batch.get_overlapping_targets("seg", "label")
                                    )
        
        print(seg_clf_out["link_logits"].shape)
        print(batch.get_overlapping_targets("seg", "link").shape)

        print(seg_clf_out["link_logits"])
        print(batch.get_overlapping_targets("seg", "link"))

        link_loss = self.pointer.loss(
                                        logits = seg_clf_out["link_logits"],
                                        targets = batch.get_overlapping_targets("seg", "link")
                                    )
                
        tw = self.hps["general"]["task_weight"]
        loss = ((1 - tw) * link_loss) + ((1 - tw) * label_loss)

        return loss