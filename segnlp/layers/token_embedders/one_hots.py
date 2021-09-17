
#basics
from typing import Sequence, Union
from torch.functional import Tensor

# pytorch
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


# segnlp
from segnlp.resources.pos import spacy_pos 
from segnlp.resources.deps import spacy_dep 
from segnlp import utils


class OneHots(nn.Module):

    def __init__(self, labels:list = None):
        super().__init__()
        self.vocab = {l.lower():i for i,l in enumerate(labels)}
        self.output_size = len(self.vocab)
    

    def __encode(self, input:Sequence):
        return torch.LongTensor([self.vocab[x.lower()] for x in input])


    def forward(self, 
                input: Sequence,
                lengths : Tensor = None,
                device : Union[str, torch.device] = "cpu"
                ):
        ids = self.__encode(input)
        one_hots = F.one_hot(ids, num_classes=self.output_size)

        if lengths is not None:
            one_hots = pad_sequence(
                                    torch.split(
                                                one_hots,
                                                utils.ensure_list(lengths),
                                                ),
                                    batch_first = True,
                                    padding_value = 0
                                    )


        return one_hots.to(device)


class PosOneHots(OneHots):

    def __init__(self):
        super().__init__(labels=spacy_pos)


class DepOneHots(OneHots):

    def __init__(self):
        super().__init__(labels=spacy_dep)

 