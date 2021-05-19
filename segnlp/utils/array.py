
#basics
import numpy as np
from typing import List, Tuple, DefaultDict
from math import floor, exp
from random import random
from itertools import product, repeat, combinations
from collections import defaultdict


#pytroch
import torch
from torch import Tensor
from torch.nn import functional as F

def ensure_flat(item, mask=None):

    if not isinstance(item[0], np.int):
        item = item.flatten()

        # Ugly fix for 2d arrays with differnet lengths
        if not isinstance(item[0], np.int):
            item = np.hstack(item)
    
    if mask is not None:
        mask = ensure_flat(ensure_numpy(mask)).astype(bool)
        if mask.shape == item.shape:
            item = item[mask]
    

    return item


def zero_pad(a):
    b = np.zeros([len(a),len(max(a,key = lambda x: len(x)))])
    for i,j in enumerate(a):
        b[i][0:len(j)] = j
    return b


def string_pad(a, dtype="<U30"):
    b = np.zeros([len(a),len(max(a,key = lambda x: len(x)))]).astype(dtype)
    b[:] = ""
    for i,j in enumerate(a):
        b[i][0:len(j)] = j
    return b


def ensure_numpy(item):

    if torch.is_tensor(item):
        item = item.cpu().detach().numpy()

    if type(item) is not np.ndarray:
        item = np.array(item)

    return item


def to_tensor(item, dtype=torch.float):
    try:
        return torch.tensor(item, dtype=dtype)
    except ValueError as e:
        return item
    except TypeError as e:
        return item


def one_hots(a):
    m = np.zeros((a.shape[0], a.shape[0]))
    m[np.arange(a.shape[0]),a] = 1
    return m



def tensor_dtype(numpy_dtype):

    if numpy_dtype == np.uint8:
        return torch.uint8

    if numpy_dtype == np.float or numpy_dtype == np.float32:
        return torch.float

    if numpy_dtype == np.int:
        return torch.long

    if numpy_dtype == np.bool:
        return torch.bool



def flatten(a):
    if isinstance(a, list):
        return [e for sublist in a for e in sublist]
    else:
        return a.flatten()

def dynamic_update(src, v, pad_value=0): 

    a = np.array(list(src.shape[1:]))
    b = np.array(list(v.shape))
    new_shape = np.maximum(a, b)
    new_src = np.full((src.shape[0]+1, *new_shape), pad_value, dtype=src.dtype)

    if len(v.shape) > 2:
        new_src[:src.shape[0],:src.shape[1], :src.shape[2]] = src
        new_src[src.shape[0],:v.shape[0], :v.shape[1]] = v
    #if len(v.shape) == 2:
    #    new_src[:src.shape[0],:src.shape[1], :] = src
    #    new_src[src.shape[0],:v.shape[0], :] = v
    else:
        new_src[:src.shape[0],:src.shape[1]] = src
        new_src[src.shape[0],:v.shape[0]] = v

    return new_src


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
        Example when we need to do this is when we have a matrix of word
        embeddings for paragraphs (batch_size, nr_paragraphs, nr_tok, word_emb)
        and we want to get spans of the paragraphs based on a mask.
        We can then use the mask which signify the spans to turn everything we
        dont want to zeros
    """
    og_shape = matrix.shape

    # # flatten to 2D
    matrix_f = torch.flatten(matrix, end_dim=-2)

    # # 2 flatten mask to 2D, last value needs to be [0] or [1]
    mask_f = mask.view((np.prod([matrix.shape[:-1]]), 1))

    # # 3 turn embs into 0 where mask is 0
    matrix_f_masked = matrix_f * mask_f

    # # 4 return to original shape
    masked_matrix = matrix_f_masked.view(og_shape)

    return masked_matrix


def create_mask(lengths, as_bool=True, flat=False):
    """
        >> torch.arange(max_len)
        tensor([0, 1, 2, 3, 4, 5])
        
        >> torch.arange(max_len).expand(len(lengths), max_len)
        tensor([[0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5]])
        >> torch.arange(max_len).expand(len(lengths), max_len)  < lengths.unsqueeze(1)
        tensor([[ True,  True, False, False, False, False],
                [ True,  True,  True,  True, False, False],
                [ True,  True, False, False, False, False],
                [ True,  True,  True,  True,  True,  True]])
        Creates a 2D range then checks for each rows where the values are smaller than the length
    """

    if not torch.is_tensor(lengths):
        lengths = torch.tensor(lengths)

    max_len = torch.max(lengths)
    mask = torch.arange(max_len).expand(len(lengths), max_len)  < lengths.unsqueeze(1)
    
    if not as_bool:
        mask = mask.type(torch.uint8)

    if flat:
        mask = mask.view(-1)      
    
    return mask


def get_all_possible_pairs(
                            start: List[List[int]],
                            end: List[List[int]],
                            device=torch.device,
                            bidir=False,
                        ) -> DefaultDict[str, List[List[Tuple[int]]]]:

    if bidir:
        get_pairs = lambda x, r: list(product(x, repeat=r))  # noqa
    else:
        get_pairs = lambda x, r: sorted(  # noqa
            list(combinations(x, r=r)) + [(e, e) for e in x],
            key=lambda u: (u[0], u[1]))

    all_possible_pairs = defaultdict(lambda: [])
    for idx_start, idx_end in zip(start, end):
        idxs = list(get_pairs(range(len(idx_start)), 2))
        if idxs:
            p1, p2 = zip(*idxs)  # pairs' unit id
            p1 = torch.tensor(p1, dtype=torch.long, device=device)
            p2 = torch.tensor(p2, dtype=torch.long, device=device)
            # pairs start and end token id.
            start = get_pairs(idx_start, 2)  # type: List[Tuple[int]]
            end = get_pairs(idx_end, 2)  # type: List[Tuple[int]]
            lens = len(start)  # number of pairs' units len  # type: int

        else:
            p1 = torch.empty(0, dtype=torch.long, device=device)
            p2 = torch.empty(0, dtype=torch.long, device=device)
            start = []
            end = []
            lens = 0

        all_possible_pairs["idx"].append(idxs)
        all_possible_pairs["p1"].append(p1)
        all_possible_pairs["p2"].append(p2)
        all_possible_pairs["start"].append(start)
        all_possible_pairs["end"].append(end)
        all_possible_pairs["lengths"].append(lens)

    all_possible_pairs["lengths"] = torch.tensor(all_possible_pairs["lengths"],
                                                 dtype=torch.long,
                                                 device=device)
    all_possible_pairs["total_pairs"] = sum(all_possible_pairs["lengths"])
    return all_possible_pairs



def util_one_hot(matrix: Tensor, mask: Tensor, num_classes: int):
    # check padding = -1
    thematrix = matrix.clone()  # avoid possible changing of the original Tensor
    pad_emb = thematrix[~mask.type(torch.bool)]
    if torch.all(pad_emb == -1):
        # change padding = 0
        pad_emb = ~mask.type(torch.bool) * 1
        thematrix += pad_emb

    return F.one_hot(thematrix, num_classes=num_classes)


def scatter_repeat(
                    src:torch.tensor,
                    value: torch.tensor, 
                    lengths:torch.tensor, 
                    length_mask:torch.BoolTensor, 
                    ):
    """
    
    scatter_repeat will shatter repeats of the src values into a new tensor.
    for example:
        if
            src = [1,2,3]
            lengths = [3,2,5,1,2]
            length_mask = [1,0,1,0,1]
        function returns
        src = [1,1,1,0,0,2,2,2,2,2,0,3,3]
        what we do is we take the values in the src, place them (scatter them) in a larger tensor,
        then repeat each value in the larger tensor by a specified length
        
        use case if for exampel if you want the scores for segments in a document to be scattered over the tokens
        which the segment comprise of, but keep the values for tokens that are not apart of segments to 0
        TODO: add fill value?
    Parameters
    ----------
    src : torch.tensor
        [description]
    lengths : torch.tensor
        [description]
    length_mask : torch.tensor
        [description]
    max_length : int
        [description]
    Returns
    -------
    [type]
        [description]
    """
    src[length_mask] = value
    repeated_src = torch.repeat_interleave(src, lengths, dim=0)
    return repeated_src


def cumsum_zero(input:torch.tensor):
    """
    torch.cumsum([4,5,10]) -> [4,9,19]
    cumsum_zero([4,5,10]) -> [0,4,19]
    """
    return torch.cat((torch.zeros(1),torch.cumsum(input, dim=0)))[:-1].type(torch.LongTensor)


def index_select_array(input:torch.tensor, index:torch.tensor):
    """
    given a input 3d tensor and a 1d index tensor selects an array at 
    dim == -1 according to index
    
    selecting works like following.
        input[i][index[i]]
    Example:
        input = [
                    [
                        [0.1,0.1],
                        [0.2,0.2],
                        [0.3,0.3],
                    ],
                    [
                        [0.4,0.4],
                        [0.5,0.5],
                        [0.6,0.6],
                    ]
                ]
    
    index = [2,1]
    returns  [
                [0.3,0.3],
                [0.5,0.5],
            ]
    """
    index_idxes = torch.full((index.shape[0],),index.shape[0],  dtype=torch.uint8)
    flat_idx = cumsum_zero(index_idxes).type(torch.LongTensor)
    flat_input = torch.flatten(input,end_dim=-2)
    return flat_input[flat_idx]


def range_3d_tensor_index(matrix: Tensor,
                          start: Tensor,
                          end: Tensor,
                          pair_batch_num: Tensor,
                          reduce_: str = "none") -> Tensor:

    # to avoid bugs, if there is a sample that does not have a unit the
    # corresponding len should be zero in batch_lens.
    batch_size = matrix.size(0)
    dim_1_size = matrix.size(1)
    new_size = (batch_size, dim_1_size)
    shape_ = len(matrix.size())

    reduce_fn = reduce_ in ["none", "mean", "sum"]
    # assertion messages:
    reduce_msg = f"Function \"{reduce_}\" is not a supported."
    num_msg = "Wrong number of pairs per sample is provided. "
    num_msg += f"Provided {len(pair_batch_num)}, expected {batch_size}."
    assert reduce_fn, reduce_msg
    assert batch_size == len(pair_batch_num), num_msg
    assert shape_ == 3, f"Wrong matrix shape, provided {shape_}, expected 3."

    # change matrix to be 2d matrix (dim0*dim1, dim2)
    mat = matrix.clone().contiguous().view(-1, matrix.size(-1))

    # construct array of indices for dimesion 0, repeating batch_id
    span_len = end - start
    idx_0 = torch.repeat_interleave(torch.arange(batch_size),
                                    pair_batch_num.cpu())
    idx_0 = torch.repeat_interleave(idx_0, span_len.cpu())

    # construct array of indices for dimesion 1
    idx_1 = torch.cat(list(map(torch.arange, start, end)))

    # Converts idx_0 and idx_1 into an array of indices suitable for the
    # converted 2d tensor
    idx_0_2d = np.ravel_multi_index([idx_0, idx_1], new_size)

    # index 2d tensor using idx_0_2d
    mat = torch.split(mat[idx_0_2d, :], span_len.tolist())
    if reduce_ == "mean":
        mat = torch.stack(list(map(torch.mean, mat, repeat(0))))
    elif reduce_ == "sum":
        mat = torch.stack(list(map(torch.sum, mat, repeat(0))))

    return mat




# def index_4D(a: torch.tensor, index: torch.tensor):
#     """
#     a is 4D tensors
#     index is 3D tensor

#     index will select values/vectors

#     """
#     b = torch.zeros((a.shape[0], a.shape[1], a.shape[-1]))
#     for i in range(index.shape[0]):
#         for j, k in enumerate(index[i]):
#             b[i][j] = a[i][j][k]
#     return b

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



# def range_3d_tensor_index(matrix: Tensor,
#                           start: List[int],
#                           end: List[int],
#                           pair_batch_num: List[int],
#                           reduce_: str = "none") -> Tensor:

#     # to avoid bugs, if there is a sample that does not have a unit the
#     # corresponding len should be zero in batch_lens.
#     batch_size = matrix.size(0)
#     dim_1_size = matrix.size(1)
#     new_size = (batch_size, dim_1_size)
#     shape_ = len(matrix.size())

#     reduce_fn = reduce_ in ["none", "mean", "sum"]
#     # assertion messages:
#     reduce_msg = f"Function \"{reduce_}\" is not a supported."
#     num_msg = "Wrong number of pairs per sample is provided. "
#     num_msg += f"Provided {len(pair_batch_num)}, expected {batch_size}."
#     assert reduce_fn, reduce_msg
#     assert batch_size == len(pair_batch_num), num_msg
#     assert shape_ == 3, f"Wrong matrix shape, provided {shape_}, expected 3."

#     # change matrix to be 2d matrix (dim0*dim1, dim2)
#     mat = matrix.clone().contiguous().view(-1, matrix.size(-1))

#     # construct array of indices for dimesion 0, repeating batch_id
#     span_len = np.array(end) - np.array(start)
#     idx_0 = np.repeat(np.arange(batch_size), pair_batch_num)
#     idx_0 = np.repeat(idx_0, span_len)

#     # construct array of indices for dimesion 1
#     idx_1 = np.hstack(list(map(np.arange, start, end)))

#     # Converts idx_0 and idx_1 into an array of indices suitable for the
#     # converted 2d tensor
#     idx_0_2d = np.ravel_multi_index(np.array([idx_0, idx_1]), new_size)

#     # index 2d tensor using idx_0_2d
#     mat = torch.split(mat[idx_0_2d, :], span_len.tolist())
#     if reduce_ == "mean":
#         mat = torch.stack(list(map(torch.mean, mat, repeat(0))))
#     elif reduce_ == "sum":
#         mat = torch.stack(list(map(torch.sum, mat, repeat(0))))

#     return mat




# def unfold_matrix(matrix_to_fold: Tensor, start_idx: List[int],
#                   end_idx: List[int], class_num_betch: List[int],
#                   fold_dim: int) -> Tensor:

#     batch_size = matrix_to_fold.size(0)
#     # construct array of indices for dimesion 0, repeating batch_id
#     span_len = np.array(end_idx) - np.array(start_idx)
#     idx_0 = np.repeat(np.arange(batch_size), class_num_betch)
#     idx_0 = np.repeat(idx_0, span_len)

#     # construct array of indices for dimesion 1
#     idx_1 = np.hstack(list(map(np.arange, start_idx, end_idx)))

#     # Get unit id for each start, end token
#     unit_id = np.hstack(list(map(np.arange, repeat(0), class_num_betch)))
#     unit_id = np.repeat(unit_id, span_len)

#     # construct the folded matrix
#     if len(matrix_to_fold.size()) > 2:
#         size = [batch_size, fold_dim, matrix_to_fold.size(-1)]
#     else:
#         size = [batch_size, fold_dim]
#     matrix = matrix_to_fold.new_zeros(size=size)

#     # fill matrix


#     matrix[idx_0, idx_1] = matrix_to_fold[idx_0, unit_id]

#     return matrix





#     # repeated_src = torch.repeat_interleave(src, lengths[length_mask], dim=0)

#     # print(repeated_src)

#     # index_ranges = torch.split(torch.arange(max_length),lengths)
#     # index = index_ranges[length_mask].view(-1)

#     # print(index)

#     # # if we have a logits and not predictions
#     # if len(src.shape) == 2:
#     #     index = torch.repeat_interleave(index.unsqueeze(1), repeats=src.shape[-1], dim=1)
#     #     shape = (
#     #             max_length,
#     #             src.shape[-1]
#     #             )
#     # else:
#     #     shape = (
#     #             max_length,
#     #             )
    
#     # print(index)
#     # scattered = torch.zeros(shape, dtype=repeated_src.dtype).scatter_(0, index, repeated_src)
#     # return scattered
