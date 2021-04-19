
from segnlp.preprocessing.encoders.base import Encoder
from typing import List, Union, Dict

class PosEncoder(Encoder):
    """
    POS tags can be found here 
    https://spacy.io/api/annotation#pos-tagging

    or by:

    >>import spacy
    >>nlp = spacy.load('en_core_web_sm')
    >>print(nlp.get_pipe("tagger").labels)

    """

    def __init__(self):
        self._name = "pos"
        pos_labels = [ 
                        '$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'ADD', 'AFX', 'CC', 'CD', 
                        'DT', 'EX', 'FW', 'HYPH', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NFP', 
                        'NN','NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 
                        'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 
                        'WDT', 'WP', 'WP$', 'WRB', 'XX', '_SP', '``'
                    ]
        self.id2label = dict(enumerate(pos_labels))
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
        return self.label2id[label]


    def decode(self, i):
        return self.id2label[i]
