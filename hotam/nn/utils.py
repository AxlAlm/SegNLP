import torch
import numpy as np

def masked_mean(m, mask):
    """
    means the rows in a given matrix based on the masked values.

    example:

    if v == m[1] and mask[i] == [0,1,1,1,0]
    mask_mean(v) = mean(v[1:4])

    Parameters
    ----------
    m : tensor
        matrix to be averaged
    mask : mask
        matrix of where m[i,j] = [0,1]

    Returns
    -------
    
        [description]
    """
    m_sum = torch.sum(m, dim=2)
    mask_sum = torch.sum(mask, dim=2, keepdim=True)
    masked_mean = m_sum / mask_sum
    masked_mean[masked_mean != masked_mean] = 0.0
    return masked_mean



def multiply_mask_matrix(matrix, mask):
    """
        Tedious way of multiplying a 4D matrix with a 3D mask.

        example when we need to do this is when we have a matrix of word embeddings 
        for paragraphs  (batch_size, nr_paragraphs, nr_tok, word_emb) and we want 
        to get spans of the paragraphs based on a mask. 

        We can then use the mask which signify the spans to turn everything we dont
        want to zeros
    """
    og_shape = matrix.shape

    ## flatten to 2D
    matrix_f = torch.flatten(matrix, end_dim=-2)
    
    ## 2 flatten mask to 2D, last value needs to be [0] or [1]
    mask_f = mask.view((np.prod([matrix.shape[:-1]]), 1))

    ## 3 turn embs into 0 where mask is 0
    matrix_f_masked = matrix_f * mask_f

    ## 4 return to original shape
    masked_matrix = matrix_f_masked.view(og_shape)

    return masked_matrix
    



def agg_emb(m, lengths, span_indexes, mode="average"):

    if mode == "mix":
        feature_dim = m.shape[-1]*3
    else:
        feature_dim = m.shape[-1]

    batch_size = m.shape[0]
    device = m.device
    agg_m = torch.zeros(batch_size, torch.max(lengths), feature_dim, device=device)

    for i in range(batch_size):
        for j in range(lengths[i]):
            ii,jj = span_indexes[i][j]

            if mode == "average":
                agg_m[i][j] = torch.mean(m[i][ii:jj], dim=0)

            elif mode == "max":
                v,_ = torch.max(m[i][ii:jj])
                agg_m[i][j] = v

            elif mode == "min":
                v,_ = torch.max(m[i][ii:jj])
                agg_m[i][j] = v

            elif mode == "mix":
                _min, _ = torch.min(m[i][ii:jj],dim=0) 
                _max, _ = torch.max(m[i][ii:jj], dim=0)  
                _mean = torch.mean(m[i][ii:jj], dim=0)     

                agg_m[i][j] = torch.cat((_min, _max, _mean), dim=0)

            else:
                raise RuntimeError(f"'{mode}' is not a supported mode, chose 'min', 'max','mean' or 'mix'")  

    return agg_m


# def reduce_and_remove(matrix, mask):

#     """
#     Given a 4D matrix turn it into a 3D by removing 3D dimension while perserving padding.

#     (similar to pytroch.utils.pad_packed_sequences, sort of)


#     for example:
#         given a 4D matrix where dims are (batch_size, nr_paragraphs, nr_spans, feature_dim),
#         if we want to get all words for all paragraphs we need to remove spans and remove padding tokens.
#         we cannot remove all values == n as padding for nr words in paragraphs needs to exists.
#         So, we need to find out max paragraph length, remove all zeros between that length and then after.
        
#         Given (batch_size, nr_paragraphs, nr_spans, nr_tokens) we get 
#     """
#     batch_size, _, _, feature_dim = matrix.shape

#     matrix_f = matrix[mask]
#     lengths = torch.sum(torch.sum(mask, dim=-1),dim=1)
    
#     new_tensor = torch.zeros((batch_size, max(lengths), feature_dim))

#     start_idx = 0
#     for end_idx in lengths:
#         new_tensor[:end_idx] = matrix_f[start_idx:start_idx+end_idx]
#         start_idx += end_idx

#     return new_tensor