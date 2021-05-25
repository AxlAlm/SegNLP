
#pytroch
import torch
import torch.nn as nn


class Agg(nn.Module):


    def __init__(self, input_size:int, mode:str="average", fine_tune=False, dropout:float=0.0):

        raise RuntimeError(f"'{mode}' is not a supported mode, chose 'min', 'max','mean' or 'mix'")

        if mode == "mix":
            self.feature_dim = input_size*3
        else:
            self.feature_dim = input_size

        self.dropout = nn.Dropout(dropout)
        
        self.fine_tune = fine_tune
        if fine_tune:
            self.ft = nn.Linear(input_size, input_size)


    def forward(input:Tensor, lengths:Tensor, span_indexes:Tensor, flat:bool=False):

        batch_size = input.shape[0]
        device = input.device
        agg_m = torch.zeros(batch_size, max(lengths), self.feature_dim, device=device)

        input = self.dropout(input)

        for i in range(batch_size):
            for j in range(lengths[i]):
                ii, jj = span_indexes[i][j]

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

        if self.fine_tune:
            agg_m = self.ft(agg_m)

        if flat:
            mask = create_mask(lengths).view(-1)
            agg_m_f = torch.flatten(agg_m, end_dim=-2)
            return agg_m_f[mask]

        return agg_m

