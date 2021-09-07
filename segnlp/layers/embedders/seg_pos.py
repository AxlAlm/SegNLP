
#pytorch
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

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
                document_paragraph_id: Tensor, 
                nr_paragraphs_doc: Tensor,
                lengths: Tensor,
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

        mask = utils.create_mask(lengths)

        #flatten and remove padding
        document_paragraph_id = document_paragraph_id[mask]
        nr_paragraphs_doc = nr_paragraphs_doc[mask]

        vec = torch.zeros((len(document_paragraph_id), 4))

        # if the segment is the first segment in the paragraph
        _, para_seg_lengths = torch.unique_consecutive(document_paragraph_id, return_counts = True)
        idxs = utils.cumsum_zero(para_seg_lengths)
        vec[:,0][idxs] = 1

        # so its not len() but index
        nr_paragraphs_doc -= 1 

        # if the segment is in  the opening paragraph
        vec[document_paragraph_id == 0, 1] += 1

        # if the segment is in the body paragraphs
        cond = torch.logical_and(document_paragraph_id > 0, document_paragraph_id < nr_paragraphs_doc)
        vec[cond, 2] += 1

        # if the segment is in the closing paragraph
        vec[document_paragraph_id == nr_paragraphs_doc, 3] += 1


        vec = pad_sequence(
                            torch.split(vec, utils.ensure_list(lengths)), 
                            batch_first=True,
                            padding_value=0
                            ).type(dtype=torch.float)

        return vec