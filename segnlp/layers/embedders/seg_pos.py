
#pytorch
import torch
import torch.nn as nn
from torch import Tensor

#segnlp
from segnlp import utils 


class SegPos(nn.Module):

    """
    Creates a vector representing a sentences position in the text.

    original texts are needed to create this feature

    feature is specifically implemented for replication of Joint Pointer NN from :
    https://arxiv.org/pdf/1612.08994.pdf

    "(3) Structural features:  Whether or not the AC is the first AC in a paragraph, 
    and Whether the AC  is  in  an  opening,  body,  or  closing  paragraph."

    we represent this info as a one hot encodings of dim==4
    """

    def __init__(self):
        super().__init__()
        self.output_size = 4


    def forward(self,
                paragraph_doc_ids: Tensor, 
                max_paragraphs: Tensor
                ):
        """
        Extracts positional features for segments in samples from:

        one hot encodig where dimN == {0,1}, dim size = 4, 
        dim0 = 1 if seg is first seg in a paragraph else 0  
        dim1 = 1 if seg is in an opening paragraph else 0
        dim2 = 1 if seg is in body (middle paragraphs) else 0
        dim3 = 1 if seg is in a closing paragraph else 0


        NOTE! in the above mentioned paper the input is paragrahs. The feature would also work document level

        """

        vec = torch.zeros((len(paragraph_doc_ids), 4))

        # if the segment is the first segment in the paragraph
        _, para_seg_lengths = torch.unique_consecutive(paragraph_doc_ids, return_counts = True)
        idxs = utils.cumsum_zero(para_seg_lengths)
        vec[:,0][idxs] = 1

        # so its not len() but index
        max_paragraphs -= 1 

        # if the segment is in  the opening paragraph
        vec[:,1][paragraph_doc_ids == 0]

        # if the segment is in the body paragraphs
        vec[:,2][paragraph_doc_ids > 0 and paragraph_doc_ids < max_paragraphs]

        # if the segment is in the closing paragraph
        vec[:,3][paragraph_doc_ids == max_paragraphs]


        return torch.LongTensor(vec, device = paragraph_doc_ids.device)