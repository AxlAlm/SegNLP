



"""
Dep labels can be found here
https://spacy.io/api/annotation

"""

spacy_dep = [
                          "","acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr", "aux",
                        "auxpass", "case", "cc", "ccomp", "compound", "conj", "cop", "csubj",
                        "csubjpass", "dative", "dep", "det","dobj", "expl", "intj", "mark", "meta",
                        "neg", "nn", "nounmod", "npmod", "nsubj", "nsubjpass", "nummod", "oprd",
                        "obj", "obl", "parataxis", "pcomp", "pobj", "poss", "preconj", "prep",
                        "prt", "punct", "quantmod", "relcl", "root", "xcomp", 

                        # Following are not listed on SpaCy but still gets annotated so have added these to the list
                        "nmod","npadvmod","subtok", "predet",
                        ]


universal_dep = [
                            "", "acl", "advcl", "advmod", "amod", "appos", "aux", "case",
                            "cc","ccomp","clf","compound", "conj", "cop", "csubj", "dep",	 
                            "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith",
                            "iobj", "list", "mark", "nmod", "nsubj", "nummod", "obj", "obl", "orphan",
                            "parataxis", "punct", "reparandum", "root", "vocative","xcomp"
                        ]