
#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F

#hotam
from hotam.nn.layers.lstm import LSTM_LAYER
from hotam.nn.layers.link_layers import Pointer

from hotam.nn.utils import agg_emb




class Encoder(nn.Module):

    def __init__(   
                    self,  
                    input_size:int, 
                    input_layer_dim:int, 
                    hidden_size:int, 
                    num_layers:int, 
                    bidirectional:int,
                    dropout:float=None,
                    ):
        super().__init__()

        self.input_layer = nn.Linear(input_size, input_layer_dim)

        self.lstm =  LSTM_LAYER(  
                                input_size=input_layer_dim,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                bidirectional=bidirectional,
                                )
        
        self.use_dropout = False
        if dropout:
            self.dropout = nn.Dropout(dropout)
            self.use_dropout = True


    def forward(self, X, lengths):

        X = self.input_layer(X)
        dense_out = F.sigmoid(X)

        out, hidden = self.lstm(dense_out, lengths)

        if self.use_dropout:
            out = self.dropout(out)

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

    2)  pass output of 1) to a Bi-LSTM. We will use the encoder outputs as representations
        for each argument component and the last states to pass to the decoder

        (output are the concatenate hidden outputs for each timestep)


    Decoder:
    ______
    
    As the decoder is working over timesteps we will use a LstmCell which we pass the last cell states to, along with appropriately 
    modified input

    For each timestep (max seq length):

    4)  Decoder takes the last states from the encoder to init the decoder.
        NOTE as the encoder is Bi-Directional we cannot just pass on the states
        from the encoder to the decoder. What do we pass on?

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
    
    def __init__(self, hyperparamaters:dict, task_dims:dict, feature_dims:dict, train_mode:bool):
        super().__init__()
        self.train_mode = train_mode
        self.OPT = hyperparamaters["optimizer"]
        self.LR = hyperparamaters["lr"]
        self.ENCODER_INPUT_DIM = hyperparamaters["encoder_input_dim"]
        self.ENCODER_HIDDEN_DIM = hyperparamaters["encoder_hidden_dim"]
        self.DECODER_HIDDEN_DIM = hyperparamaters["decoder_hidden_dim"]
        self.ENCODER_NUM_LAYERS = hyperparamaters["encoder_num_layers"]
        self.ENCODER_BIDIR = hyperparamaters["encoder_bidir"]
            
        self.F_DROPOUT = hyperparamaters["feature_dropout"]
        self.ENC_DROPOUT = hyperparamaters["encoder_dropout"]
        self.DEC_DROPOUT = hyperparamaters["decoder_dropout"]

        self.FEATURE_DIM = feature_dims["doc_embs"]

        # α∈[0,1], will specify how much to weight the two task in the loss function
        self.TASK_WEIGHT = hyperparamaters["task_weight"]


        if self.DECODER_HIDDEN_DIM != self.ENCODER_HIDDEN_DIM*self.ENCODER_NUM_LAYERS:
            raise RuntimeError("Encoder - Decoder dimension missmatch. As the decoder is initialized by the encoder states the decoder dimenstion has to be encoder_dim * num_encoder_layers")
        
        self.use_feature_dropout = False
        if self.F_DROPOUT:
            self.use_feature_dropout = True
            self.feature_dropout = nn.Dropout(self.F_DROPOUT)

        self.encoder = Encoder(
                                input_size=self.FEATURE_DIM,
                                input_layer_dim=self.ENCODER_INPUT_DIM,
                                hidden_size=self.ENCODER_HIDDEN_DIM,
                                num_layers= self.ENCODER_NUM_LAYERS,
                                bidirectional=self.ENCODER_BIDIR,
                                dropout = self.ENC_DROPOUT
                                )

        self.decoder = Pointer(
                                input_size=self.DECODER_HIDDEN_DIM,
                                hidden_size=self.DECODER_HIDDEN_DIM,
                                dropout = self.DEC_DROPOUT
                                )


        self.ac_clf_layer = nn.Linear(self.ENCODER_HIDDEN_DIM*(2 if self.ENCODER_BIDIR else 1), task_dims["ac"])
        self.loss = nn.NLLLoss(reduction="sum", ignore_index=-1)


    @classmethod
    def name(self):
        return "JointPN"


    def forward(self, batch):
                        
        unit_embs = agg_emb(batch["token"]["word_embs"], 
                            lengths = batch["unit"]["lengths"],
                            span_indexes = batch["unit"]["span_idxs"], 
                            mode = "average"
                            )
        
        #combining features
        X = torch.cat((unit_embs, batch["unit"]["doc_embs"]), dim=-1)
        
        if self.use_feature_dropout:
            X = self.feature_dropout(X)

        # 1-2 | Encoder
        # encoder_output = (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM*LAYER*DIRECTION)
        encoder_out, (encoder_h_s, encoder_c_s) = self.encoder(X, batch["unit"]["lengths"])

        # 3-7 | Decoder
        # (BATCH_SIZE, SEQ_LEN, SEQ_LEN)
        # prob distributions (softmax)
        pointer_probs = self.decoder(encoder_out, encoder_h_s, encoder_c_s, batch["unit"]["mask"])

        label_probs =  F.softmax(self.ac_clf_layer(encoder_out),dim=-1)

        if self.train_mode:
            # we want to ignore -1  in the loss function so we set pad_values to -1, default is 0
            batch.change_pad_value(-1)

            pointer_probs_2d = torch.flatten(pointer_probs, end_dim=-2)
            link_loss = self.loss(torch.log(pointer_probs_2d), batch["relation"].view(-1))

            label_probs_2d = torch.flatten(ac_probs, end_dim=-2)
            label_loss = self.loss(torch.log(ac_probs_2d), batch["ac"].view(-1))

            
            total_loss = ((1-self.TASK_WEIGHT) * link_loss) + ((1-self.TASK_WEIGHT) * label_loss)

            output.add_loss(task="total",       data=total_loss)
            output.add_loss(task="link",        data=link_loss)
            output.add_loss(task="label",       data=label_loss)


        label_preds = torch.argmax(label_probs,  dim=-1)
        link_preds = torch.argmax(pointer_probs, dim=-1)

        output.add_preds(task="label",          level="unit", data=label_preds)
        output.add_preds(task="link",           level="unit", data=link_preds)


