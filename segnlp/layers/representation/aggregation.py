

class Aggregate(nn.Module):


    def __init__(self, input_shape, mode:str="average", fine_tune=False):
        if mode == "mix":
            self.feature_dim = input_shape*3
        else:
            self.feature_dim = input_shape
        
        self.fine_tune = fine_tune
        if fine_tune:
            self.ft = nn.Linear(input_shape, input_shape)


    def forward(m, lengths, span_indexes, flat:bool=False):

        batch_size = m.shape[0]
        device = m.device
        agg_m = torch.zeros(batch_size, max(lengths), self.feature_dim, device=device)

        for i in range(batch_size):
            for j in range(lengths[i]):
                ii, jj = span_indexes[i][j]

                if self.mode == "average":
                    agg_m[i][j] = torch.mean(m[i][ii:jj], dim=0)

                elif self.mode == "max":
                    v, _ = torch.max(m[i][ii:jj])
                    agg_m[i][j] = v

                elif self.mode == "min":
                    v, _ = torch.max(m[i][ii:jj])
                    agg_m[i][j] = v

                elif self.mode == "mix":
                    _min, _ = torch.min(m[i][ii:jj],dim=0)
                    _max, _ = torch.max(m[i][ii:jj], dim=0)
                    _mean = torch.mean(m[i][ii:jj], dim=0)

                    agg_m[i][j] = torch.cat((_min, _max, _mean), dim=0)

                else:
                    raise RuntimeError(f"'{mode}' is not a supported mode, chose 'min', 'max','mean' or 'mix'")


        if self.fine_tune:
            agg_m = self.ft(agg_m)

        if flat:
            mask = create_mask(lengths).view(-1)
            agg_m_f = torch.flatten(agg_m, end_dim=-2)
            return agg_m_f[mask]

        return agg_m

