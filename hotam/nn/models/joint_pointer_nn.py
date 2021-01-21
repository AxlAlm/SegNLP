
#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F

#hotam
from hotam.nn.layers.lstm import LSTM_LAYER
from hotam.nn.layers.attention import CONTENT_BASED_ATTENTION
from hotam.nn.utils import masked_mean, multiply_mask_matrix


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
    
    As the decoder is working over timesteps the LSTM will be a LstmCell to 
    which we manually pass the last cell states to, along with appropriately 
    modified input

    For each timestep in max sequence length:

    4)  Decoder takes the last states from the encoder to init the decoder.
        NOTE as the encoder is Bi-Directional we cannot just pass on the states
        from the encoder to the decoder. What do we pass on?

        a) concatenation of forward layer outputs. This means that the decoder hidden is
        always encoder hidden * 2.  In the paper they set the Decoder Hidden to 512,
        and the encoder hidden to 256, which fit with the rule.

        c)  foward and backwards concate of last layer. Fit with the hyperparamaters
            in the paper.

        b)  last foward layer outputs. Will not work with the hyperparamaters given in the paper.
            i.e. Encoder will output 256 dims, but the Decoder and attention will then not match as
            they are 512


        As input, we pass and empty zero tensor as there is no input arrow
        architecture in the paper. This also make sense as there are no previous 
        decoding timesteps, which the input is intended to be, hence there is 
        nothing to pass
    
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
    
    def __init__(self, hyperparamaters:dict, task2labels:dict, feature2dim:dict):
        super().__init__()
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

        self.FEATURE_DIM = feature2dim["word_embs"] + 2 # FOR DOC POS

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

        self.decoder = Decoder(
                                input_size=self.DECODER_HIDDEN_DIM,
                                hidden_size=self.DECODER_HIDDEN_DIM,
                                dropout = self.DEC_DROPOUT
                                )


        nr_ac_labels = len(task2labels["ac"])
        self.ac_clf_layer = nn.Linear(self.ENCODER_HIDDEN_DIM*(2 if self.ENCODER_BIDIR else 1), nr_ac_labels)

        self.loss = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)


    @classmethod
    def name(self):
        return "JointPN"


    def forward(self, batch):
        
        adu_embs = batch["word_embs"]
                
        #4D mask, mask over the words in each ac in each input
        ac_token_mask = batch["ac_token_mask"]
        
        #3D mask, mask over acs in each input
        ac_mask = batch["ac_mask"]

        lengths = batch["lengths_seq"]

        # turn all work embeddigns that are not ACs to 0s (e.g. all words beloning to Argument Markers are turn to 0)
        masked_ac_word_embs =  multiply_mask_matrix(adu_embs, ac_token_mask)

        # aggregating word embeddings while taking masked values into account
        agg_ac_embs = masked_mean(masked_ac_word_embs, ac_token_mask)
        
        #combining features
        X = torch.cat((agg_ac_embs, batch["doc_embs"]), dim=-1)
        
        if self.use_feature_dropout:
            X = self.feature_dropout(X)

        # 1-2 | Encoder
        # encoder_output = (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM*LAYER*DIRECTION)
        encoder_out, (encoder_h_s, encoder_c_s) = self.encoder(X, lengths)

        # 3-7 | Decoder
        # (BATCH_SIZE, SEQ_LEN, SEQ_LEN)
        pointer_probs = self.decoder(encoder_out, encoder_h_s, encoder_c_s, ac_mask)

        # we want to ignore -1  in the loss function so we set pad_values to -1, default is 0
        batch.change_pad_value(-1)

        #8
        #(BATCH_SIZE * SEQ_LEN, NUM_LABELS)   
        pointer_probs_2d = torch.flatten(pointer_probs, end_dim=-2)

        relation_loss = self.TASK_WEIGHT * self.loss(pointer_probs_2d, batch["relation"].view(-1))
        relation_preds = torch.argmax(pointer_probs,dim=-1)

        #9
        #(BATCH_SIZE, SEQ_LEN, NUM_LABELS)
        ac_probs =  F.softmax(self.ac_clf_layer(encoder_out),dim=-1)

        #(BATCH_SIZE * SEQ_LEN, NUM_LABELS)    
        ac_probs_2d = torch.flatten(ac_probs, end_dim=-2)
        ac_loss = (1-self.TASK_WEIGHT) * self.loss(ac_probs_2d, batch["ac"].view(-1))
        ac_preds = torch.argmax(ac_probs, dim=-1)

        #10
        total_loss = relation_loss + ac_loss

        return {    
                    "loss": {   
                                "total": total_loss,
                                "ac": ac_loss, 
                                "relation": relation_loss,
                                }, 
                    "preds": {
                                "ac": ac_preds, 
                                "relation": relation_preds,
                            },
                    "probs": {}
                }


class Decoder(nn.Module):


    def __init__(self, input_size:int, hidden_size:int, dropout=None):
        super().__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.lstm_cell =  nn.LSTMCell(input_size, hidden_size)

        self.attention = CONTENT_BASED_ATTENTION(
                                                    input_dim=hidden_size,
                                                    )
        
        self.use_dropout = False
        if dropout:
            self.dropout = nn.Dropout(dropout)
            self.use_dropout = True


    def forward(self, encoder_outputs, encoder_h_s, encoder_c_s, mask):

        seq_len = encoder_outputs.shape[1]
        batch_size = encoder_outputs.shape[0]

        #we concatenate the forward direction lstm states
        # from (NUM_LAYER*DIRECTIONS, BATCH_SIZE, HIDDEN_SIZE) ->
        # (BATCH_SIZE, HIDDEN_SIZE*NUM_LAYER)
        layer_dir_idx = list(range(0,encoder_h_s.shape[0],2))
        encoder_h_s = torch.cat([*encoder_h_s[layer_dir_idx]],dim=1)
        encoder_c_s = torch.cat([*encoder_c_s[layer_dir_idx]],dim=1)

        decoder_input = torch.zeros(encoder_h_s.shape)
        prev_h_s = encoder_h_s
        prev_c_s = encoder_c_s

        #(BATCH_SIZE, SEQ_LEN)
        pointer_probs = torch.zeros(batch_size, seq_len, seq_len)
        for i in range(seq_len):
            
            prev_h_s, prev_c_s = self.lstm_cell(decoder_input, (prev_h_s, prev_c_s))

            if self.use_dropout:
                prev_h_s = self.dropout(prev_h_s)

            decoder_input = self.input_layer(decoder_input)
            decoder_input = F.sigmoid(decoder_input)

            pointer_softmax = self.attention(prev_h_s, encoder_outputs, mask)
        
            pointer_probs[:, i] = pointer_softmax
            
        return pointer_probs


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