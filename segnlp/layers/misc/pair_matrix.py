



class PairMatrix(nn.Module):


    def __init__(max_units:int, modes=["cat", "mean"], rel_pos=False):
        self.modes = modes
        self.max_units = max_units
        self.rel_pos = rel_pos


    def forward(input_emb,  pair_mask:torch.Tensor=None):
        
        device = input_emb.device
        batch_size = input_emb.shape[0]
        dim1 = input_emb.shape[1]

        shape = (batch_size, dim1, dim1, input_emb.shape[-1])
        m = torch.reshape(torch.repeat_interleave(input_emb, dim1, dim=1), shape)
        mT = m.transpose(2, 1)

        to_cat = []
        if "cat" in modes:
            to_cat.append(m)
            to_cat.append(mT)
        
        if "multi" in modes:
            to_cat.append(m*mT)

        if "mean" in modes:
            to_cat.append((m+mT /2))

        if "sum" in modes:
            to_cat.append(m+mT)
        

        #adding one_hot encoding for the relative position
        if self.rel_pos:
            one_hot_dim = (self.max_units*2)-1
            one_hots = torch.tensor(
                                        [
                                        np.diag(np.ones(one_hot_dim),i)[:dim1,:one_hot_dim] 
                                        for i in range(dim1-1, -1, -1)
                                        ], 
                                        dtype=torch.uint8,
                                        device=device
                                        )
            one_hots = one_hots.repeat_interleave(batch_size, dim=0)
            one_hots = one_hots.view((batch_size, dim1, dim1, one_hot_dim))
            
            to_cat.append(one_hots)

        pair_matrix = torch.cat(to_cat, axis=-1)

        if pair_mask is not None:
            pairs_flat = torch.flatten(pair_matrix, end_dim=-2)
            return pairs_flat[pair_mask]
        else:
            return pair_matrix