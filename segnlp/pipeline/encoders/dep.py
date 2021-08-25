
from .base import Encoder
from typing import List, Union, Dict


class DepEncoder(Encoder):

    """
    
    Dep labels can be found here
    https://spacy.io/api/annotation#dependency-parsing

    """

    def __init__(self):
        self._name = "dep"
        dep_labels_eng = [
                          "","acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr", "aux",
                        "auxpass", "case", "cc", "ccomp", "compound", "conj", "cop", "csubj",
                        "csubjpass", "dative", "dep", "det","dobj", "expl", "intj", "mark", "meta",
                        "neg", "nn", "nounmod", "npmod", "nsubj", "nsubjpass", "nummod", "oprd",
                        "obj", "obl", "parataxis", "pcomp", "pobj", "poss", "preconj", "prep",
                        "prt", "punct", "quantmod", "relcl", "root", "xcomp", "csubj", "nmod",
                        ## ADDED:
                        "npadvmod","subtok", "predet",
                        ]
        # universal_deps = [
        #                     "", "acl", "advcl", "advmod", "amod", "appos", "aux", "case",
        #                     "cc","ccomp","clf","compound", "conj", "cop", "csubj", "dep",	 
        #                     "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith",
        #                     "iobj", "list", "mark", "nmod", "nsubj", "nummod", "obj", "obl", "orphan",
        #                     "parataxis", "punct", "reparandum", "root", "vocative","xcomp"
        #                 ]
        self.id2label = dict(enumerate(dep_labels_eng))
        self.label2id = {l:i for i,l in self.id2label.items()}
        

    def __len__(self):
        return len(self.id2label)
    
    @property
    def keys(self):
        return list(self.label2id.keys())

    @property
    def name(self):
        return self._name


    def encode(self, label):
        return self.label2id[label.lower()]


    def decode(self, i):
        return self.id2label[i]
