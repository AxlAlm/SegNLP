
#pytroch
import torch
import torch.nn as nn
from torch import Tensor

class Agg(nn.Module):


    def __init__(self, input_size:int, mode:str="mean", dropout:float=0.0):
        super().__init__()

        supported_modes = set(['min', 'max','mean', 'mix'])

        if mode not in supported_modes:
            raise RuntimeError(f"'{mode}' is not a supported mode, chose 'min', 'max','mean' or 'mix'")

        self.mode = mode

        if self.mode == "mix":
            self.output_size = input_size*3
        else:
            self.output_size = input_size

        self.dropout = nn.Dropout(dropout)


    def forward(self, input:Tensor, lengths:Tensor, span_idxs:Tensor):

        batch_size = input.shape[0]
        device = input.device
        agg_m = torch.zeros(batch_size, max(lengths), self.output_size, device=device)

        input = self.dropout(input)

        for i in range(batch_size):
            for j in range(lengths[i]):
                ii, jj = span_idxs[i][j]

                if ii == 0 and jj == 0:
                    continue

                #when slicing we need to add 1 to the roof so we dont miss the last token
                jj += 1

                if self.mode == "average":
                    agg_m[i][j] = torch.mean(input[i][ii:jj], dim=0)

                elif self.mode == "max":
                    v, _ = torch.max(input[i][ii:jj])
                    agg_m[i][j] = v

                elif self.mode == "min":
                    v, _ = torch.max(input[i][ii:jj])
                    agg_m[i][j] = v

                elif self.mode == "mix":
                    _min, _ = torch.min(input[i][ii:jj],dim=0)
                    _max, _ = torch.max(input[i][ii:jj], dim=0)
                    _mean = torch.mean(input[i][ii:jj], dim=0)

                    agg_m[i][j] = torch.cat((_min, _max, _mean), dim=0)

        # if flat:
        #     mask = create_mask(lengths).view(-1)
        #     agg_m_f = torch.flatten(agg_m, end_dim=-2)
        #     return agg_m_f[mask]

        return agg_m

