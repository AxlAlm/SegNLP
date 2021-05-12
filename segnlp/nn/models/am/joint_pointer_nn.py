
#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F

#segnlp
from segnlp.nn.layers.rep_layers import LSTM
from segnlp.nn.layers.link_layers import Pointer
from segnlp.nn.utils import agg_emb


class Encoder(nn.Module):

    def __init__(   
                    self,  
                    input_size:int, 
                    hidden_size:int, 
                    num_layers:int, 
                    bidirectional:int,
                    dropout:float=None,
                    ):
        super().__init__()
        self.input_layer = nn.Linear(input_size, input_size)


        self.lstm =  LSTM(  
                                input_size=input_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bidirectional=bidirectional,
                                )
        
        self.use_dropout = False
        if dropout:
            self.dropout = nn.Dropout(dropout)
            self.use_dropout = True


    def forward(self, X, lengths):

        if self.use_dropout:
            X = self.dropout(X)

        dense_out = torch.sigmoid(self.input_layer(X))
        out, hidden = self.lstm(dense_out, lengths)

        return out, hidden



class JointPN(nn.Module):

    """
    
    Paper:
    https://arxiv.org/pdf/1612.08994.pdf


    more on Pointer Networks:
    https://arxiv.org/pdf/1409.0473.pdf

    https://papers.nips.cc/paper/5866-pointer-networks.pdf  


    A quick read:
    https://medium.com/@sharaf/a-paper-a-day-11-pointer-networks-59f7af1a611c


    NN FLow:

    Encoder:
    ______
    1) pass input to a fully-connected layer with sigmoid activation

    2)  pass output of 1) to a Bi-LSTM. 
  

    Decoder:
    ______
    
    As the decoder is working over timesteps we will use a LstmCell which we pass the last cell states to, along with appropriately 
    modified input

    For each timestep (max seq length):

    4)  Decoder takes the last states (cell and timestep) from the encoder to init the decoder.

        foward and backwards concatenations of last layer in encoder lstm. 

        As first input, we pass and zero tensor as there is no input arrow
        architecture in the paper. This also make sense as there are no previous 
        decoding timesteps, which the input is intended to be, hence there is 
        nothing to pass. One could pass a random value tensor representing START
    
    5)  the input to the next decoder is set as the hidden state outputed from the
        LSTM-cell. Hidden state and cells state are also set as next states for 
        the cell in the next timestep (just as an LSTM)

    6)  the hidden state is then passed to a Linear Layer with sigmoid activation (FC3 in the paper)
        NOTE! This layer is meant to modify the input prior to the decoder lstm but as we see in the 
        figure 3 in the paper there is no input to the first decoder step, which means we can just set 
        this layer after and apply it to the next input to the next decoder step, which will
        be in accordance with figure 3 in the paper.
    
    7)  the hidden state of the decoder is then passed to Content Based Attention layer along
        with all the Encoder outputs. We then "compare" the decoder output at timestep i (di) with
        all the encoder outputs so that we can get a probability that di is pointing to any En
    
    ________

    8) to get the relation predictions we simply take the argmax of the attention output which is softmax.
        to get the loss we take the sum of the log softmax  * task weight.

    9) to predict Argument Component we pass the encoder output to a linear layer and apply softmax
        to get probs, and argmax for predictions. For loss we use log_softmax * task weight.

    10) Total loss of the task is relation loss + ac loss



    NOTE! Regarding dropout. 
    In the paper they state they are using dropout but not where they are applying it 
    so we can only guess where the dropout is applied. In this implementation we have hence
    decided to divide dropout into 3 types feature_dropout (applied at features), 
    encoder dropout (applied on out LSTM), and decoder dropout (applied on out LSTM)

    """
    
    def __init__(self, hyperparamaters:dict, task_dims:dict, feature_dims:dict, inference:bool):
        super().__init__()
        self.inference = inference
        self.OPT = hyperparamaters["optimizer"]
        self.LR = hyperparamaters["lr"]
        self.ENCODER_HIDDEN_DIM = hyperparamaters["encoder_hidden_dim"]
        self.DECODER_HIDDEN_DIM = hyperparamaters["decoder_hidden_dim"]
        self.ENCODER_NUM_LAYERS = hyperparamaters["encoder_num_layers"]
        self.ENCODER_BIDIR = hyperparamaters["encoder_bidir"]
            
        self.ENC_DROPOUT = hyperparamaters["encoder_dropout"]
        self.DEC_DROPOUT = hyperparamaters["decoder_dropout"]

        # times 3 becasue we use the max+min+avrg embeddings
        self.FEATURE_DIM =  feature_dims["doc_embs"] + (feature_dims["word_embs"] * 3)

        # α∈[0,1], will specify how much to weight the two task in the loss function
        self.TASK_WEIGHT = hyperparamaters["task_weight"]

        if self.DECODER_HIDDEN_DIM != self.ENCODER_HIDDEN_DIM*(2 if self.ENCODER_BIDIR else 1):
            raise RuntimeError("Encoder - Decoder dimension missmatch. As the decoder is initialized by the encoder states the decoder dimenstion has to be encoder_dim * nr_directions")
        

        self.encoder = Encoder(
                                input_size=self.FEATURE_DIM,
                                hidden_size=self.ENCODER_HIDDEN_DIM,
                                num_layers= self.ENCODER_NUM_LAYERS,
                                bidirectional=self.ENCODER_BIDIR,
                                dropout = self.ENC_DROPOUT
                                )

        self.decoder = Pointer(
                                input_size=self.FEATURE_DIM,
                                hidden_size=self.DECODER_HIDDEN_DIM,
                                dropout = self.DEC_DROPOUT
                                )


        self.label_clf = nn.Linear(self.ENCODER_HIDDEN_DIM*(2 if self.ENCODER_BIDIR else 1), task_dims["label"])
        self.loss = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)


    @classmethod
    def name(self):
        return "JointPN"


    def forward(self, batch, output):

        unit_embs = agg_emb(batch["token"]["word_embs"], 
                            lengths = batch["unit"]["lengths"],
                            span_indexes = batch["unit"]["span_idxs"], 
                            mode = "mix"
                            )
        
        X = torch.cat((unit_embs, batch["unit"]["doc_embs"]), dim=-1)

        # 1-2 | Encoder
        # encoder_output = (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM*LAYER*DIRECTION)
        encoder_out, (encoder_h_s, encoder_c_s) = self.encoder(X, batch["unit"]["lengths"])

        # 3-7 | Decoder
        # We get the last hidden cell states and timesteps and concatenate them for each directions
        # from (NUM_LAYER*DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE) -> (BATCH_SIZE, HIDDEN_SIZE*NR_DIRECTIONS)
        # -2 gets the last layer thats forward and -1 get the lasy layer backwards
        encoder_h_s = torch.cat((encoder_h_s[-2], encoder_h_s[-1]), dim=1)
        encoder_c_s = torch.cat((encoder_c_s[-2], encoder_c_s[-1]), dim=1)

        # OUTPUT: (BATCH_SIZE, SEQ_LEN, SEQ_LEN)
        link_logits = self.decoder(
                                    X,
                                    encoder_out, 
                                    encoder_h_s, 
                                    encoder_c_s, 
                                    batch["unit"]["mask"], 
                                    return_softmax=False
                                    )

        # OUTPUT: (BATCH_SIZE, SEQ_LEN, nr_labels)
        label_logits =  self.label_clf(encoder_out)

        if not self.inference:

            link_loss = self.loss(torch.flatten(link_logits, end_dim=-2), batch["unit"]["link"].view(-1))
            label_loss = self.loss(torch.flatten(label_logits, end_dim=-2), batch["unit"]["label"].view(-1))
            
            total_loss = ((1-self.TASK_WEIGHT) * link_loss) + ((1-self.TASK_WEIGHT) * label_loss)

            output.add_loss(task="total",       data=total_loss)
            output.add_loss(task="link",        data=link_loss)
            output.add_loss(task="label",       data=label_loss)


        label_preds = torch.argmax(label_logits,  dim=-1)
        link_preds = torch.argmax(link_logits, dim=-1)

        #print(link_preds[3], batch["unit"]["link"][3])

        output.add_preds(task="label",          level="unit", data=label_preds)
        output.add_preds(task="link",           level="unit", data=link_preds)

        return output

