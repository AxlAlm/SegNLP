#basic
import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from tqdm import tqdm
from string import punctuation
import warnings
import spacy
from pprint import pprint

# networkx
import networkx as nx
from networkx import Graph as nxGraph

from networkx.algorithms.tree.recognition import is_tree

# DGL
import dgl
from dgl import DGLGraph
from dgl.traversal import topological_nodes_generator as traverse_topo

#spacy
from spacy.language import Language


#nltk
import nltk

#segnlp
from segnlp.utils import timer
from segnlp import get_logger
logger = get_logger("PREPROCESSOR")

#import multiprocessing as mp

# punctuation fixes
punctuation += "’‘,`'" + '"'
strange_characters = {
                        '``':'"',
                        "''":'"'
                    }


# nltk download checking
# # TODO: move this to a better spot and maybe not where its called upon importing.
# try:
#     nltk.data.find('tokenizers/punkt')
#     nltk.download('tokenizers/averaged_perceptron_tagger')
# except LookupError:
#     nltk.download("punkt")
#     nltk.download('averaged_perceptron_tagger')


class TextProcessor:


    def _init_text_processor(self):

        # storing the current row for each level, used to fetch ids etc for lower lever data
        self._level_row_cache : dict = {}
        self._nlp_name = "Spacy"
        self.nlp : Language = self._load_nlp_model()

        # Text Processing
        self.level2parents : dict = {
                                "token": ["sentence", "paragraph", "document"],
                                "sentence": ["paragraph", "document"],
                                "paragraph": ["document"],
                                "document": []
                                }

        self.parent2children : dict = {
                                "document": ["paragraph","sentence", "token"],
                                "paragraph": ["sentence", "token"],
                                "sentence": ["token"],
                                "token": []
                                }
        self._prune_hiers()



    def __paragraph_tokenize(self, doc:str) -> List[str]:
        """ Tokenizes document into paragraphs by splitting by new line ("\n"). 
    
        Will ignore any empty strings

        Parameters
        ----------
        doc : str
            a document to be tokenized into paragraphs

        Returns
        -------
        List[str]
            list of paragraphs
        """
        return [p for p in doc.split("\n") if p]



    def __get_global_id(self, level:str) -> int:
        """Creates a global id for the current level, .e.g. sentence, word, paragraph.

        An ID is equal to a counter, i.e. sentence nr 10 will have ID 0, sentence 11 ID 10 and so on.


        Parameters
        ----------
        level : str
            the level of the unit to be processed, e.g. "sentence", "document", "token" etc

        Returns
        -------
        int
            ID for the current unit on the given level
        """
        try:
            return self._level_row_cache[level]["id"] + 1
            #return self.stack_level_data[level][-1]["id"] + 1
        except KeyError as e:
            return 0


    def __get_parent_text(self, level:str) -> Tuple[str,str]:
        """        Gets the text of last added parent to the given child-level. This function is used
        to simple fetch the last parent and the top parent processed so it can be processed on a lower level.

        E.g. if param level = token, function will fetch the last sentences processed and top parent 
        (if there is one) e.g paragraph or document.

        Parameters
        ----------
        level : str
            the level of the unit to be processed, e.g. "sentence", "document", "token" etc

        Returns
        -------
        Tuple[str,str]
            The top parent text and the closest parent text
            
        """
        parents = self.level2parents[level]
        closest_parent_text = self._level_row_cache[parents[0]]["str"]
        top_parent_text = self._level_row_cache[parents[-1]]["str"]
        return top_parent_text, closest_parent_text


    def __get_char_idx(self, level:str) -> int:
        """Gets the current character index of the text. Given a level (e.g. "sentence", "token" etc) 
        the fucntion will return an integer represting the index.

        if the level stack (e.g. no previously processed data on this level) the current index is set to 0, as its the first

        if the last row added for the given level belongs to a different text sample (fetched based on the dataset_level, 
        e.g. on which level the top parent text is on), we set the current character index to 0 as we are begining on a new sample


        Parameters
        ----------
        level : str
            the level of the unit to be processed, e.g. "sentence", "document", "token" etc


        Returns
        -------
        int
            current character index with the sample text
        """
        
        parents = self.level2parents[level]
        #top_parent_row = self.stack_level_data[parents[-1]][-1]
        top_parent_row = self._level_row_cache[parents[-1]]

        #if not self.stack_level_data[level]:
        if level not in self._level_row_cache:
            return  0
        
        level_row = self._level_row_cache[level]
        #level_row = self.stack_level_data[level][-1]

        if level_row[f"{self.input_level}_id"] != top_parent_row["id"]:
            current_idx = 0
        else:
            current_idx = level_row["char_end"]

        return current_idx


    def __get_structure_info(self, level:str) -> Dict[str,int]:
        """Provides the ids for the parents for last unit on the given level. This is needed so we know which parent is a perent for 
        any given child and vice versa.


        For example: If we pass "token" as level and our samples (e.g. dataset_level) are documents
        we get the id for the document, the paragraph and the sentence that the last token belongs to.


        Parameters
        ----------
        level : str
            the level of the unit to be processed, e.g. "sentence", "document", "token" etc

        Returns
        -------
        Dict[str,int]
            keys identify the level (e.g. "sentence_id") and int is the id
        """

    
        parents = self.level2parents[level]
        first_parent = parents[0]
        upper_parents = parents[1:]
        last_parent_row = self._level_row_cache[first_parent]
        #last_parent_row = self.stack_level_data[first_parent][-1]

        #inherent all the ids of the closest parent
        info = {k:v for k,v in last_parent_row.items() if "id" in k or "nr" in k}
        info[f"{first_parent}_id"] = info.pop("id")


        # if there ARE previous units of the given level, e.g. the stack
        # for tokens is NOT empty we know the unit is not the first of its kind
        first = True
        current_id_in_parent = -1 # <-- -1 because we are adding 1 later in the the __build_X functions.
        #if self.stack_level_data[level]:
        if level in self._level_row_cache:
            # last_level_row = self.stack_level_data[level][-1]
            last_level_row = self._level_row_cache[level]
            first = False

        for parent in upper_parents:
            parent_level_id = f"{parent}_{level}_id"
            parent_id = f"{parent}_id"

            # if the current unit is not last
            if not first:

                # if the last unit of the given unit has a different parent id,
                # we know can set the current_id_in_parent to -1, as we know its the first 
                # unit that will be in that parent unit. I.e. no unit on the same level have the same parent id.
                if last_level_row[parent_id] != info[parent_id]:
                    current_id_in_parent = -1 # <-- -1 because we are adding 1 later in the the __build_X functions.
                else:
                    current_id_in_parent = last_level_row[parent_level_id]
            
            info[parent_level_id] = current_id_in_parent


        return info


    def __build_tokens(self, spacy_sentence):
        """tokenizes and pos-tags a sentence to create rows/processed units for tokens. Adds the processed tokens 
        the token level stack. (see class description)

        for any token in the sentence we create an global id, an id for the token in the sentence, set character span
        add pos tag and create a label slot. Then we add all the parent ids.

        example:
                {
                    "id": 24 <--- global id, token in corpus
                    "sentence_token_id": 2, <--- which token in the sentence
                    "char_start": 10,
                    "char_end": 16,
                    "str": "giraff",
                    "label": None,
                    "pos": "NN",
                    "sentence_id": 150,
                    "paragraph_id": 70,
                    "document_id": 50,
                    "paragraph_sentence_id": 2, <--- which sentence in the paragraph
                    "document_sentence_id:" 3, <--- which sentence in the document
                    "paragraph_document_id" 1, <--- which paragraph in the document
                    "document_token_id": 20,  <--- which token in the document
                    "paragraph_token_id" 4, <--- which token in the paragraph

                    }

        
        for example, given the document_sentence_id,sentence_token_id and document_id  we know that the token is 
        the 3rd token in the 4th sentence in document 50.


        """

        parent_info = self.__get_structure_info("token")
        doc, sentence = self.__get_parent_text("token")
        current_token_idx = self.__get_char_idx("token")

        # we create a mapping of all tokens indexes. Then when we encounter a token that we need to remove
        # we update the portion after the token we removed so the indexes after fill in the cap of the removed token
        # e.g. we change indexes so the final version of normalized_indexes contain a range between two number if we ignore 
        # the indexes of ignored tokens. 
        # We need this for depheads for example.
        #                                example = [1,2,2,3,4,5,6,6,7,8,9] 
        #                                           1,2,3,4,5,6,7,8,9,10,11 # original
        #                                           1   2,3,4,5,6,  7,8,9   # selected
        #
        normalized_indexes = np.arange(len(spacy_sentence))

        #i = 0
        #for tok in spacy_sentence:
        for i, tok in enumerate(spacy_sentence):
            token = tok.text

            if "\t" in token or "\n" in token:
                normalized_indexes = np.concatenate((normalized_indexes[:i],normalized_indexes[i:]-1))
                continue

            token_len = len(token)
            token = strange_characters.get(token, token)

            if current_token_idx == 0:
                start = 0
            else:
                # -2 here is just to start searching from a few characters back 
                # it allows the tokenizer idxes to be a bit off and we will still find the correct
                # char index in the origial text. 
                start = doc.find(token, max(0, current_token_idx-2))
            
            end = start + token_len
            dephead = tok.head.i - spacy_sentence.start
            
            row =  {
                    "id": self.__get_global_id("token"),
                    "sentence_token_id": normalized_indexes[i],
                    "char_start": start,
                    "char_end": end,
                    "str": token.lower(),
                    "pos": tok.tag_,
                    "dephead": normalized_indexes[dephead],
                    "deprel": tok.dep_
                    #
                    }

            for parent in self.level2parents["token"][1:]:
                parent_info[f"{parent}_token_id"] += 1

            row.update(parent_info)

            self._level_row_cache["token"] = row
            self.__data_stack.append(row)

            current_token_idx = end

            #i += 1

        # stanza_doc = self.nlp(sentence)
        # spacy_doc = self.nlp(sentence)
        # s_toks = [t.text for t in stanza_doc.iter_tokens()]
        
        # tokens = nltk.word_tokenize(sentence)
        # token_pos = nltk.pos_tag(tokens)
        # for i, (token,pos) in enumerate(token_pos):
        #     token_len = len(token)
        #     token = strange_characters.get(token, token)

        #     if current_token_idx == 0:
        #         start = 0
        #     else:
        #         start = doc.find(token, max(0, current_token_idx-2))
        #     end = start + token_len

        #     row =  {
        #             "id": self.__get_global_id("token"),
        #             "sentence_token_id": i,
        #             "char_start": start,
        #             "char_end": end,
        #             "str": token.lower(),
        #             #"pos": token_dict["xpos"],
        #             #"dephead": token_dict["head"],
        #             #"deprel": token_dict["deprel"]
        #             #
        #             }

        #     for parent in self.level2parents["token"][1:]:
        #         parent_info[f"{parent}_token_id"] += 1

        #     row.update(parent_info)

        #     self._level_row_cache["token"] = row

        #     self._data_stack.append(row)

        #     current_token_idx = end


    def __build_sentences(self):
        """
        Sentence tokenize text (assumed to be a paragraph) and creates 
        a row/processed unit for each sentence. For each sentence call __build_tokens().

        For each sentence we set a global id, character span and the ids of all parents.

        See __build_tokens() documentation for a more detailed example of format.
        """

        parent_info = self.__get_structure_info("sentence")
        doc, paragraph = self.__get_parent_text("sentence")
        current_sent_idx = self.__get_char_idx("sentence")
        #paragraph, current_sent_id,  current_sent_idx = self.__get_text_id_idx("sentence")

        spacy_doc = self.nlp(paragraph)
        

        removed_sents = 0
        for i, sentence in enumerate(spacy_doc.sents):

            if len(str(sentence).strip()) == 0:
                removed_sents += 1
                continue
        
            if current_sent_idx == 0:
                start_idx = 0
            else:
                start_idx = doc.find(str(sentence), current_sent_idx) #, current_sent_idx)

            end_idx = start_idx + len(str(sentence)) #sentence[-1].idx + len(sentence[-1])
            row =  {
                    "id": self.__get_global_id("sentence"),
                    "paragraph_sentence_id": i - removed_sents,
                    "str":str(sentence),
                    "char_start": start_idx,
                    "char_end": end_idx,
                    }

            for parent in self.level2parents["sentence"][1:]:
                parent_info[f"{parent}_sentence_id"] += 1

            row.update(parent_info)

            self._level_row_cache["sentence"] = row
            current_sent_idx = end_idx
            self.__build_tokens(sentence)


    def __build_paragraphs(self):
        """
        Tokenize text int paragraphs (assumed to be a documents) and creates 
        a row/processed unit for each paragraph.  For each sentence call __build_sentences().

        For each par we set a global id, character span and the ids of all parents.

        See __build_tokens() documentation for a more detailed example of format.
        """

        parent_info = self.__get_structure_info("paragraph")
        doc, _ = self.__get_parent_text("paragraph")
        current_para_idx = self.__get_char_idx("paragraph")

        paras = self.__paragraph_tokenize(doc)

        nr_paras = len(paras)
        for i,para in enumerate(paras):
            para_len = len(para)

            if i == 0:
                start = 0
            else:
                start = doc.find(para, current_para_idx)

            end = start + para_len
 
            para_len = len(para)
            row =  {
                    "id": self.__get_global_id("paragraph"),
                    "document_paragraph_id": i,
                    "str": para,
                    "char_start": start,
                    "char_end": end,
                    "nr_paragraphs_doc":nr_paras,
                    }

            for parent in self.level2parents["paragraph"][1:]:
                parent_info[f"{parent}_paragraph_id"] += 1

            row.update(parent_info)

            #self.stack_level_data["paragraph"].append(row)
            self._level_row_cache["paragraph"] = row
            current_para_idx = end

            self.__build_sentences()


    def _prune_hiers(self):
        """
        Given the dataset level we set the hierarchy. For now the heirarchy dicts, found in the Datset Class,
        are containing the full supported hierarchiy; from document to tokens. But if our datset level is for
        example  sentences we need to remove documents and paragraphs for the assumed hierarchy so we know that 
        what to process.
        """

        new_level2parents = {self.input_level:[]}
        for k, v in self.level2parents.items():

            if k in self.parent2children[self.input_level]:
                new_level2parents[k] = v[:v.index(self.input_level)+1]
        
        self.level2parents = new_level2parents


    def _load_nlp_model(self):
        return spacy.load("en_core_web_lg", disable=["ner"])


    def __fix_deps(self, df:pd.DataFrame):

        """

        If sample level is not sentences we need to correct the Dependecy Parsing as its
        done on sentence level.  To get a full graph for each sample we set the ROOT of each sentence to 
        point to the ROOT in the previous sentence.        
        """

        df["root_idx"] = None


        for _, sample in df.groupby(f"{self.sample_level}_id", sort=False):
            
            sentences  = sample.groupby("sentence_id", sort=False)

            current_position = 0
            depheads = []
            sent_roots = []
            for _, sent_df in sentences:
                
                # normalize the indexes of of the depheads to be on a sample level, e.g. take into acount
                # the previous sentences
                sent_depheads = sent_df["dephead"].to_numpy() + current_position
                deprel_set = set(sent_df["deprel"].to_list())


                # n = list(range(len(sent_depheads)))
                # e = sent_df["dephead"].to_list()
                # sent_graph = dgl.graph((n, e)).to_networkx().to_undirected()
                # if not nx.is_connected(sent_graph):
                #     pprint(list(zip(n, e, sent_df["str"], sent_df["deprel"].to_list())))
                #     print(LOOOL)

                #if not "ROOT" in deprel_set:

    
                
                sent_root_idx = sent_df["deprel"].to_list().index("ROOT")

                #print(sent_root_idx)
    
                # we change the index of the HEAD of the root be the idx of the previous ROOT
                if sent_roots:
                    sent_depheads[sent_root_idx] = sent_roots[-1] 
                    sent_root_idx = sent_root_idx + current_position
                
                
                current_position += sent_df.shape[0]
                sent_roots.append(sent_root_idx)
                depheads.extend(sent_depheads)
            

            #print(depheads)
            df.loc[sample.index, "dephead"] = depheads


            # the root of the sample will be the ROOT of the first sentence
            df.loc[sample.index, "root_idx"] = [sent_roots[0]] * len(sample)

        df["root_idx"] = df["root_idx"].to_numpy(int)

        return df


    def _process_text(self, doc:str):
        """given a text string processes it appropriately. Meaning, if the string is a document
        we will process the document into document, paragraphs, documents and tokens. Creating 
        dataframes for each of the levels. 

        Parameters
        ----------
        string : str
            text sample to be processed
        text_type : str
            type of the given string
        text_id : str
            id for the given string

        Raises
        ------
        NotImplementedError
            if a text type that is not of the valid ones an error will be thrown. Text type 
            has to be either document, paragraph or sentence.
        """

        if self.input_level in self._level_row_cache:
            text_id = self._level_row_cache[self.input_level]["id"] + 1
        else:
            text_id = 0

        self._level_row_cache[self.input_level] = {
                                                    "id":text_id,
                                                    "str":doc
                                                    }
        self.__data_stack = []
        if self.input_level == "document":
            self.__build_paragraphs()
        elif self.input_level == "paragraph":
            self.__build_sentences()      
        elif self.input_level == "sentence":
            self.__build_tokens(next(self.nlp(doc).sents))   
        else:
            raise NotImplementedError(f'"{self.input_level}" is not a understood type')

        df = pd.DataFrame(self.__data_stack)

        if self.sample_level != "sentence":
            df = self.__fix_deps(df)

        return df







    # def __infer_text_type(self, text:str) -> str:
    #     """infer the unit type of the text naively. 

    #     if the text contains \n its a document
    #     else if the text contains more than 2 fullstops its a paragraph
    #     else if the text contain spaces its a sentence
    #     else its a token

    #     Parameters
    #     ----------
    #     text : str
    #         text

    #     Returns
    #     -------
    #     str
    #         text type
    #     """
    #     if "\n" in text.strip():
    #         t =  "doc"
    #     elif text.count(".") >= 2:
    #         t = "paragraph"
    #     elif " " in text.strip():
    #         t = "sentence"
    #     else:
    #         t = "token"
        
    #     warnings.warn(f"No text_type was passed. text_type infered to '{t}'")
    #     return t
