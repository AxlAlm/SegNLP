   
   
"""
POS tags can be found here 
https://spacy.io/api/annotation

or by:

>>import spacy
>>nlp = spacy.load('en_core_web_sm')
>>print(nlp.get_pipe("tagger").labels)

"""
# pytroch
import torch

# segnlp
from .vocab import Vocab


class Pos(Vocab):

    def __init__(self):
        pos =   [ 
                    '$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'ADD', 'AFX', 'CC', 'CD', 
                    'DT', 'EX', 'FW', 'HYPH', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NFP', 
                    'NN','NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 
                    'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 
                    'WDT', 'WP', 'WP$', 'WRB', 'XX', '_SP', '``'
                ]
        super().__init__(vocab = pos)