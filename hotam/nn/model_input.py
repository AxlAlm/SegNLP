

import numpy as np
from collections import defaultdict 


#pytroch 
import torch

#hotam 
from hotam.utils import dynamic_update, tensor_dtype
from hotam.visuals.tree_graph import TextNode, create_tree
from hotam.utils import ensure_numpy

class ModelInput(dict):


    def __init__(self, 
                label_encoders=None,
                label_colors=None,
                ):
        super().__init__()
        self.label_encoders = label_encoders
        self.label_colors = label_colors
        self.current_epoch = None
        self.pad_value = 0
        self._size = 0
        self._ids = []


    def __len__(self):
        return len(self._ids)

    @property
    def ids(self):
        return self._ids

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
    

    def to_tensor(self):

        for level in self:
            for k,v in self[level].items():

                if k == "text":
                    continue

                self[level][k] = torch.tensor(v, dtype=tensor_dtype(v.dtype))
        return self


    def to_numpy(self):

        self._ids = np.array(self._ids)
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
        #for task in self.all_tasks:
        self[level][task][self[level][task] == self.pad_value] = new_value
            # self[task][~self[f"{self.prediction_level}_mask"].type(torch.bool)] = -1
     

    def add(self, k, v, level):
        # if k not in self:
        #     self[k] = [v]
        # else:
        #     self[k].append(v)
        # if k == "id":
        #     if "id" in k:
        #         self[k] = [v]
        #     else:
        #         self[k].append(v)


        if k == "id":
            self._ids.append(v)
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
                self[level][k] = dynamic_update(self[level][k], v)


    def show_sample(self, sample_id=None):

        if sample_id is None:
            sample_id = self.ids[0]

        self.prediction_level = "token"
        
        tree_graph = True
        link_label_exist = True

        if "link" not in self[self.prediction_level]:
            tree_graph = False
    
        if "link_label" not in self[self.prediction_level]:
            link_label_exist = False


        idx = int(np.where(self.ids == sample_id)[0])

        if tree_graph:
            
            nodes = []
            start = 0
            j = 0
            for i in range(self["span"]["lengths"][idx]):
                
                length = unit_length = self["span"]["lengths_tok"][idx][i]

                if self["span"]["none_span_mask"][idx][i]:

                    link = int(self[self.prediction_level]["link"][idx][start:start+length][0])
                    label = int(self[self.prediction_level]["label"][idx][start:start+length][0])
                    label = self.label_encoders["label"].decode(label)

                    tokens = [t.decode("utf-8") for t  in self[self.prediction_level]["text"][idx][start:start+length]]
                    text  = " ".join(tokens)

                    link_label = None
                    if link_label_exist:
                        link_label = self[self.prediction_level]["link_label"][idx][start:start+length][0]
                        link_label = self.label_encoders["link_label"].decode(link_label)

                    if j == link:
                        link = "ROOT"
                    
                    nodes.append(TextNode(
                                            ID=j,
                                            link=link, 
                                            label=label,
                                            label_color=self.label_colors[label],
                                            link_label=link_label,
                                            link_label_color=self.label_colors.get(link_label, "grey"),
                                            text=text,
                                            )
                                )
                    j += 1

                start += length
            
            tree = create_tree(tree=TextNode(), nodes=nodes)
            tree.show()
                        



        # if "mask" in self["token"]:
        #     print("LINK", self["token"]["mask"][idx])

        # if "link" in self["token"]:
        #     print("LINK", self["token"]["link"][idx])

        # if "label" in self["token"]:
        #     print("LABELs", self["token"]["label"][idx])

        # if "link_label" in self["token"]:
        #     print("LINK_LABEL", self["token"]["link_label"][idx])

        # print(self["span"]["none_span_mask"][idx])
        # print(self["span"]["lengths_tok"][idx])
