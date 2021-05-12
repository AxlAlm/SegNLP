
#basics
import numpy as np

#pytroch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

#segnlp
from segnlp.nn.layers.rep_layers import CharEmb
from segnlp.nn.layers.rep_layers import LSTM
from segnlp.utils import zero_pad

# use a torch implementation of CRF
from torchcrf import CRF
#from allennlp.modules.conditional_random_field import ConditionalRandomField as CRF


class LSTM_CNN_CRF(nn.Module):

    """

    BiLSTM CNN CRF network
    
    ARGUMENT MINING PAPER
    paper1:
    https://arxiv.org/pdf/1704.06104.pdf

    ORIGINAL PAPER INTRODUCING NETWORK
    paper2:
    https://www.aclweb.org/anthology/P16-1101.pdf


    Network FLOW:

    1) get word embeddings

    2)  get character embeddings by using a CNN layer.
        character embeddings are then pooled to get a representation
        in the same dimension as words. I.e. from character embeddings
        we create a word embedding
    
    3) concatenate Word and Character-(word)-embeddings

    4) pass concatenated embeddings to bidirectional LSTM

    5) pass LSTM output to a linear layer to get logits for each class

    6) pass output to CRF

    7) voila: predictions

    
    STagT system corrections (SHOULD WE DO THIS OR NOT?):
        1) Invalid BIO structure, i.e., “I” follows “O”.

        2)  A predicted component is not homogeneous:for example, one token is predicted to link
            to the following argument component, whileanother token within the same component is
            predicted to link to the preceding argumentcomponent.

        3) A link goes ‘beyond’ the actual text, e.g.,
            when a premise is predicted to link to anothercomponent at ‘too large’ distance|d|.

    NOTE! 
    this network supports multitask learning by using seperate CRF layers per task.

    """ 

    def __init__(self, hyperparamaters:dict, task_dims:dict, feature_dims:dict, inference:bool):
        super().__init__()
        self.inference = inference
        
        self.OPT = hyperparamaters["optimizer"]
        self.LR = hyperparamaters["lr"]
        #self.DROPOUT = hyperparamaters["dropout"]
        self.HIDDEN_DIM = hyperparamaters["hidden_dim"]
        self.CHAR_DIM = hyperparamaters["char_dim"]
        self.KERNEL_SIZE = hyperparamaters["kernel_size"]
        self.NUM_LAYERS = hyperparamaters["num_layers"]
        self.BI_DIR = hyperparamaters["bidir"]
        self.N_FILTERS = hyperparamaters["n_filters"]
        self.WORD_EMB_DIM = feature_dims["word_embs"]


        self.finetune = nn.Linear(self.WORD_EMB_DIM, self.WORD_EMB_DIM)

        self.lstm = LSTM(  
                                input_size=self.WORD_EMB_DIM + self.N_FILTERS,
                                hidden_size=self.HIDDEN_DIM,
                                num_layers= self.NUM_LAYERS,
                                bidirectional=self.BI_DIR,
                                dropout=hyperparamaters["dropout"]
                                )
        
        #char_vocab = len(self.dataset.encoders["chars"])
        self.char_cnn  = CharEmb(  
                                emb_size = self.CHAR_DIM,
                                n_filters = self.N_FILTERS,
                                kernel_size = self.KERNEL_SIZE,
                                dropout=hyperparamaters["dropout"]
                                )
        self.dropout = nn.Dropout(hyperparamaters["dropout"])

        #output layers. One for each task
        self.output_layers = nn.ModuleDict()

        #Crf layers, one for each task
        self.crf_layers = nn.ModuleDict()

        for task, output_dim in task_dims.items():
            self.output_layers[task] = nn.Linear(self.HIDDEN_DIM*(2 if self.BI_DIR else 1), output_dim)
            self.crf_layers[task] = CRF(num_tags=output_dim)
            #self.crf_layers[task] = CRF(num_tags=output_dim, batch_first=True)


    @classmethod
    def name(self):
        return "LSTM_CNN_CRF"


    def forward(self, batch, output):

        lengths = batch["token"]["lengths"]
        mask = batch["token"]["mask"]

        #1
        word_embs = self.finetune(batch["token"]["word_embs"])

        #2
        char_embs = self.char_cnn(batch["token"]["chars"])

        #3
        cat_emb = torch.cat((word_embs, char_embs), dim=-1)

        #4 feed packed to lstm
        lstm_out, _ = self.lstm(cat_emb, lengths)


        lstm_out = self.dropout(lstm_out)


        for task, output_layer in self.output_layers.items():

            #5
            dense_out = output_layer(lstm_out)
            
            #6
            crf = self.crf_layers[task]

            if not self.inference:

                # crf doesnt work if padding is not a possible class, so we put padding as 0 which
                # will be default to "O" in BIO or "None" (see label encoders)
                batch.change_pad_value(level="token", task=task, new_value=0)

                target_tags = batch["token"][task]                


                loss = -crf( 
                            inputs=dense_out,
                            tags=target_tags,
                            mask=mask,
                            )

                # loss = -crf( 
                #             emissions=dense_out,
                #             tags=target_tags,
                #             mask=mask,
                #             reduction='mean'
                #             )
                                
                output.add_loss(task=task,   data=loss)

            #7
            preds = crf.viterbi_tags(
                                    logits=dense_out,
                                    mask=mask
                                    )
            preds = [p[0] for p in preds]
            #print(preds)

            # preds = crf.decode( 
            #                     emissions=dense_out, 
            #                     mask=mask
            #                     )
            
            #print(torch.tensor(zero_pad(preds), dtype=torch.long)[0],target_tags[0])
            # print(
            #         batch["token"][task], 
            #         torch.tensor(zero_pad(preds))
            #         )
            #output.add_preds(task=task, level="token", data= batch["token"][task])
            output.add_preds(task=task, level="token", data=torch.tensor(zero_pad(preds), dtype=torch.long))
            
        return output

