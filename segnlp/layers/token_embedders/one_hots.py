
#basics
from typing import Sequence, Union
from numpy.lib.arraysetops import isin
from torch.functional import Tensor

# pytorch
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


# segnlp
from segnlp import utils
from segnlp.resources import vocabs


class OneHots(nn.Module):

    def __init__(self, vocab: Union[str, list]) -> None:
        super().__init__()
        self.vocab = getattr(vocabs, vocab)() if isinstance(vocab, str) else vocab
        self.output_size = len(self.vocab)
    

    def forward(self, 
                input: Sequence,
                lengths : Tensor = None,
                device : Union[str, torch.device] = "cpu"
                ):

        ids = torch.LongTensor(self.vocab[input]).to(device)
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
 