
from hotam.preprocessing.encoders.base import Encoder
from typing import List, Union, Dict


class LabelEncoder(Encoder):


    def __init__(self, name:str, labels:list):
        self._name = name
        #self._max_sample_length = max_sample_length

        #TODO better way to do this ?
        zero_label = False
        for label in labels:

            if isinstance(label, int):
                break

            if "None" in label or "O" in label:
                zero_label = label
                break
                
        if zero_label:
            labels.remove(zero_label)
            labels.insert(0,zero_label)

        
        self.id2label = dict(enumerate(labels))
        self.label2id = {l:i for i,l in self.id2label.items()}


    def __len__(self):
        return len(self.id2label)

    @property
    def labels(self):
        return list(self.label2id.keys())

    @property
    def ids(self):
        return list(self.id2label.keys())

    @property
    def name(self):
        return self._name


    def encode(self, label):
        l = self.label2id.get(label, -1)
        return self.label2id.get(label, -1)


    def decode(self, i):
        return self.id2label.get(i, "None")
