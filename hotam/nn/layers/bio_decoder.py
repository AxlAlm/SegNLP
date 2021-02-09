
from hotam.utils import ensure_numpy, ensure_flat

import re 
import torch

#torch.nn.Module

class BIO_Decoder():


    def __init__(self, B:list, I:list, O:list, apply_correction:bool=True):
        Bs = "-|".join([str(b) for b in B]) + "-"
        Is = "-|".join([str(i) for i in I]) + "-"
        Os = "-|".join([str(o) for o in O]) + "-"
        self._Os = O

        self._apply_correction = apply_correction

        # if we have invalid BIO structure we can correct these
        #  https://arxiv.org/pdf/1704.06104.pdf, appendix
        # 1) I follows 0 -> allow OI to be interpreted as a B
        if self._apply_correction:
            Bs += f"|(?<=({Os}))({Is})"

        self.pattern = re.compile(f"({Bs})({Is})*|({Os})+")


    def _bio_decode_sample(self, encoded_bios):

        encoded_bios = ensure_flat(ensure_numpy(encoded_bios))
        encoded_bios_str = "-".join(encoded_bios.astype(str)) + "-"

        #self.__seg_id = 0
        #self.__lengths = []
        #self.__indexes = []
        self.__lengths = []
        self.__seg_types = []
        def repl(m):
            bio_list = m.group().split("-")[:-1] #when splitting on "-" we will alway create an empty "" at the end
            length = len(bio_list)

            seg_type = "AC"
            set_labels = list(set(bio_list))
            # print(set_labels[0] in self._Os, set_labels[0], self._Os, bio_list)
            if int(set_labels[0]) in self._Os:
                #seg_id_sequence  = "NONE-" * lenght
                seg_type = None
            # else:
            #     seg_id_sequence = f'{seg_type}_{self.__seg_id}-' * lenght
            #     self.__seg_id += 1

            self.__lengths.append(length)
            self.__seg_types.append(seg_type)

            #return seg_id_sequence
            #return length, seg_type
            return ""


        re.sub(self.pattern, repl, encoded_bios_str)

        lenghts = self.__lengths
        seg_types = self.__seg_types

        #marked_spans_str = re.sub(self.pattern, repl, encoded_bios_str)
        #marked_spans = marked_spans_str.split("-")[:-1]
        #lengths = self.__lengths
        #assert len(lengths) == self.__seg_id
        #assert len(marked_spans) == len(encoded_bios), f"span length: {len(marked_spans)}, bio length: {len(encoded_bios)}"

        return lenghts, seg_types


    def decode(self, batch_bios, lengths):
        
        batch_size = batch_bios.shape[0]
        batch_lengths = []
        for i in range(batch_size):
            lengths = self._bio_decode_sample(batch_bios[i][:lengths[i]])
            batch_lengths.append(lengths)

        print(batch_lengths)
        return batch_lengths



