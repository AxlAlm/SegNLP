

import numpy as np
from collections import defaultdict 


#pytroch 
import torch

#segnlp 
from segnlp.utils import dynamic_update, tensor_dtype
from segnlp.visuals.tree_graph import arrays_to_tree
from segnlp.utils import ensure_numpy

class ModelInput(dict):


    def __init__(self, 
                label_encoders=None,
                label_colors=None,
                ):
        super().__init__()
        self.label_encoders = label_encoders
        self.label_colors = label_colors
        self.current_epoch = None
        self.label_pad_value = -1
        self._size = 0
        self._idxs = []
        self.oo = []

    def __len__(self):
        return len(self._idxs)

    @property
    def idxs(self):
        return self._idxs

    @property
    def levels(self):
        return list(self.keys())


    def to(self, device):
        self.device = device
        for level in self:
            for k,v in self[level].items():
                if torch.is_tensor(v):
                    self[level][k] = v.to(self.device)
                    
        return self
    

    def to_tensor(self, device="cpu"):

        for level in self:
            for k,v in self[level].items():

                if k == "text":
                    continue

                self[level][k] = torch.tensor(v, dtype=tensor_dtype(v.dtype), device=device)
        return self


    def to_numpy(self):

        self._idxs = np.array(self._idxs)
        for level in self:

            for k,v in self[level].items():
                
                if isinstance(v, np.ndarray):
                    continue
                
                if isinstance(v[0], np.ndarray):
                    dtype = v[0].dtype
                else:
                    dtype = np.int

                self[level][k] = np.array(v, dtype=dtype)

        return self


    def change_pad_value(self, level:str, task:str, new_value:int):
        self[level][task][self[level][task] == self.label_pad_value] = new_value
     

    def add(self, k, v, level, pad_value=0):
        # if k not in self:
        #     self[k] = [v]
        # else:
        #     self[k].append(v)
        # if k == "id":
        #     if "id" in k:
        #         self[k] = [v]
        #     else:
        #         self[k].append(v)


        if k == "idxs":
            self._idxs.append(v)
            return
        
        if level not in self:
            self[level] = {}
         
        if k not in self[level]:
            if isinstance(v, np.ndarray):
                self[level][k] = np.expand_dims(v, axis=0)
            else:
                self[level][k] = [v]
        else:
            if isinstance(v, int):
                self[level][k].append(v)
            else:
                self[level][k] = dynamic_update(self[level][k], v, pad_value=pad_value)


    def sort(self):
        lengths = self["token"]["lengths"]
        
        lengths_decending = np.argsort(lengths)[::-1]

        #r = np.arange(lengths_decending.shape[0])
        #si = np.argsort(a)
        self.oo = np.argsort(lengths_decending)

        for group in self:
            for k, v in self[group].items():
                self[group][k] = v[lengths_decending]

  
    def show_sample(self, sample_id=None):

        if sample_id is None:
            sample_id = self.idxs[0]

        idx = int(np.where(self.idxs == sample_id)[0])

        self.prediction_level = "token"
        
        tree_graph = True
        link_labels = None

        if "link" not in self[self.prediction_level]:
            tree_graph = False
    
        if "link_label" in self[self.prediction_level]:
            link_labels = self.label_encoders["link_label"].decode_list(ensure_numpy(self[self.prediction_level]["link_label"][idx]))

        links = ensure_numpy(self[self.prediction_level]["link"][idx])
        labels = self.label_encoders["label"].decode_list(ensure_numpy(self[self.prediction_level]["label"][idx]))

        if tree_graph:
            arrays_to_tree(
                            ensure_numpy(self["span"]["lengths"][idx]), 
                            ensure_numpy(self["span"]["lengths_tok"][idx]),
                            ensure_numpy(self["span"]["none_span_mask"][idx]),
                            links=links,
                            labels=labels,
                            tokens=[t.decode("utf-8") for t in self[self.prediction_level]["text"][idx]],
                            label_colors=self.label_colors,
                            link_labels=link_labels
                            )




        # idx = int(np.where(self.ids == sample_id)[0])

        # if tree_graph:
            
        #     nodes = []
        #     start = 0
        #     j = 0
        #     for i in range(self["span"]["lengths"][idx]):
                
        #         length = unit_length = self["span"]["lengths_tok"][idx][i]

        #         if self["span"]["none_span_mask"][idx][i]:

        #             link = int(self[self.prediction_level]["link"][idx][start:start+length][0])
        #             label = int(self[self.prediction_level]["label"][idx][start:start+length][0])
        #             label = self.label_encoders["label"].decode(label)

        #             tokens = [t.decode("utf-8") for t  in self[self.prediction_level]["text"][idx][start:start+length]]
        #             text  = " ".join(tokens)

        #             link_label = None
        #             if link_label_exist:
        #                 link_label = self[self.prediction_level]["link_label"][idx][start:start+length][0]
        #                 link_label = self.label_encoders["link_label"].decode(link_label)

        #             if j == link:
        #                 link = "ROOT"
                    
        #             nodes.append(TextNode(
        #                                     ID=j,
        #                                     link=link, 
        #                                     label=label,
        #                                     label_color=self.label_colors[label],
        #                                     link_label=link_label,
        #                                     link_label_color=self.label_colors.get(link_label, "grey"),
        #                                     text=text,
        #                                     )
        #                         )
        #             j += 1

        #         start += length
            
        #     tree = create_tree(tree=TextNode(), nodes=nodes)
        #     tree.show()
                    