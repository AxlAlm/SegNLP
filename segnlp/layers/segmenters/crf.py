
#pytroch
import torch
import torch.nn as nn
from torch import Tensor

#segnlp
from segnlp.utils import zero_pad
from segnlp.utils import timer

# AllenNLP
from allennlp.modules.conditional_random_field import ConditionalRandomField


class CRF(nn.Module):

    """
    Simply a wrapper over the CRF implementation of AllenNLP

    source:
    https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py
 
    """

    def __init__(self, 
                input_size:int, 
                output_size:int, 
                ):
        super().__init__()
        self.clf = nn.Linear(input_size, output_size)
        self.crf = ConditionalRandomField(    
                        num_tags=output_size,
                        include_start_end_transitions = False
                        )

    @classmethod
    def name(self):
        return "CRF"


    def forward(self, input:Tensor, mask:Tensor):

        logits = self.clf(input)
        preds = self.crf.viterbi_tags(
                                logits = logits,
                                mask = mask,
                                #top_k=1
                                )
        preds = [p[0] for p in preds]
    
        preds = torch.tensor(zero_pad(preds))
        return logits, preds
    

    def loss(self, logits:Tensor, targets:Tensor, mask:Tensor):
        targets[targets == -1] = 0
        loss = -self.crf( 
                            inputs=logits,
                            tags=targets,
                            mask=mask,
                            )
        return loss

