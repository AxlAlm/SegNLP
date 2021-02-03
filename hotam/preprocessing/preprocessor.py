#basic
import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from tqdm import tqdm
from string import punctuation
import warnings
import spacy

#nltk
# import nltk

#hotam
from hotam.utils import timer
from hotam import get_logger
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


class Preprocessor:

    """
    Class for preprocessing text.


    Preprocesses text samples and structures them in a hiearchial manner by creating
    dataframes for each units/subunits of the given text sample, i.e. levels.
    The level dataframes created for e.g. a Document are:

            Document -> Paragraphs -> Sentences -> tokens

    The Dataframes contain ids of parent levels allowing you to both treverse on
    a level but also in a tree like manner.

    USEAGE:
    -------
    for preprocessing a text one use add_sample()
    for preprocessing multiple texts use add_samples()

    --
    Given a text sample Preprocessor does the following:

    ( 0) if level of the dataset samples is not given it guesses a level. E.g. if the dataset
    contain sentences the dataset level is set to "sentences")

    1) Given the level of the text sample it processes the text sample by levels

    2) appends processed level units along with ids of parents and so on to its appropriate stack. these are dicts.

    3) when preprocessing is done, clears the level stacks. This means taking the processed level untis (a list of dicts)
        createing a dataframe and then appending to the existing dataframe


    For example: if you pass a sentence, 2 dataframes will be returned. One for sentences
    and one for token where in the latter will contain the ids for the former.

    TODO:

    1) Adding choice of NLP tool. Adding support for Spacy, StandfordCORE NLP

    2) Multiprocessing

    (3 add choice to how data should be preprocessed. for example, what do do with titles?, or sub-headings?)'

    """

    def __paragraph_tokenize(self, doc:str) -> List[str]:
        """ Tokenizes document into paragraphs by splitting by new line ("\n").

        Will ignore any empty strings

        Paragraph tokenization includes title in the first paragraph (might need to be changed)

        Parameters
        ----------
        doc : str
            a document to be tokenized into paragraphs

        Returns
        -------
        List[str]
            list of paragraphs
        """

        paras = []
        stack = ""
        chars_added = 0
        for i, para in enumerate(doc.split("\n")):
            #print([para])
            #para =  para.strip()

            if not para:
                stack += "\n"
                continue

            #to add title to first paragraph
            if i == 0:
                if len(re.findall(r"[\.?!]", para)) <= 2:

                    #if para[-1] not in punctuation:
                        #para += "."

                    stack += para
                    continue

            if stack:
                para = stack  + "\n" + para
                stack = ""

            paras.append(para)

        return paras


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
        #closest_parent_text = self.stack_level_data[parents[0]][-1]["text"]
        #top_parent_text = self.stack_level_data[parents[-1]][-1]["text"]
        closest_parent_text = self._level_row_cache[parents[0]]["text"]
        top_parent_text = self._level_row_cache[parents[-1]]["text"]
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

        if level_row[f"{self.dataset_level}_id"] != top_parent_row["id"]:
            current_idx = 0
        else:
            current_idx = level_row["char_end"]

        return current_idx


    def __get_structure_ids(self, level:str) -> Dict[str,int]:
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
        ids ={k:v for k,v in last_parent_row.items() if "id" in k}
        ids[f"{first_parent}_id"] = ids.pop("id")


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
                if last_level_row[parent_id] != ids[parent_id]:
                    current_id_in_parent = -1 # <-- -1 because we are adding 1 later in the the __build_X functions.
                else:
                    current_id_in_parent = last_level_row[parent_level_id]

            ids[parent_level_id] = current_id_in_parent


        return ids


    def __infer_text_type(self, text:str) -> str:
        """infer the unit type of the text naively.

        if the text contains \n its a document
        else if the text contains more than 2 fullstops its a paragraph
        else if the text contain spaces its a sentence
        else its a token

        Parameters
        ----------
        text : str
            text

        Returns
        -------
        str
            text type
        """
        if "\n" in text.strip():
            t =  "doc"
        elif text.count(".") >= 2:
            t = "paragraph"
        elif " " in text.strip():
            t = "sentence"
        else:
            t = "token"

        warnings.warn(f"No text_type was passed. text_type infered to '{t}'")
        return t


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
                    "text": "giraff",
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

        parent_ids = self.__get_structure_ids("token")
        doc, sentence = self.__get_parent_text("token")
        current_token_idx = self.__get_char_idx("token")

        #print("HELLO DOC", doc)
        for i, tok in enumerate(spacy_sentence):
            token = tok.text
            token_len = len(token)
            token = strange_characters.get(token, token)

            if current_token_idx == 0:
                start = 0
            else:
                # -2 here is just to start searching from a few characters back
                # it allows the tokenizer idxes to be a bit off and we will still find the correct
                # char index in the origial text.
                start = doc.find(token, max(0, current_token_idx-2))

            assert tok.i - spacy_sentence.start == i
            assert tok.head.i - spacy_sentence.start >= 0
            assert tok.head.i - spacy_sentence.start < len(sentence)


            end = start + token_len
            row =  {
                    "id": self.__get_global_id("token"),
                    "sentence_token_id": i,
                    "char_start": start,
                    "char_end": end,
                    "text": token.lower(),
                    "pos": tok.tag_,
                    "dephead": tok.head.i - spacy_sentence.start,
                    "deprel": tok.dep_
                    #
                    }

            for parent in self.level2parents["token"][1:]:
                parent_ids[f"{parent}_token_id"] += 1

            row.update(parent_ids)

            self._level_row_cache["token"] = row
            self._data_stack.append(row)

            current_token_idx = end


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
        #             "text": token.lower(),
        #             #"pos": token_dict["xpos"],
        #             #"dephead": token_dict["head"],
        #             #"deprel": token_dict["deprel"]
        #             #
        #             }

        #     for parent in self.level2parents["token"][1:]:
        #         parent_ids[f"{parent}_token_id"] += 1

        #     row.update(parent_ids)

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

        parent_ids = self.__get_structure_ids("sentence")
        doc, paragraph = self.__get_parent_text("sentence")
        current_sent_idx = self.__get_char_idx("sentence")
        #paragraph, current_sent_id,  current_sent_idx = self.__get_text_id_idx("sentence")

        spacy_doc = self.nlp(paragraph)

        for i, sentence in enumerate(spacy_doc.sents):

            if current_sent_idx == 0:
                start_idx = 0
            else:
                start_idx = doc.find(str(sentence), current_sent_idx) #, current_sent_idx)

            end_idx = start_idx + len(str(sentence)) #sentence[-1].idx + len(sentence[-1])
            row =  {
                    "id": self.__get_global_id("sentence"),
                    "paragraph_sentence_id": i,
                    "text":str(sentence),
                    "char_start": start_idx,
                    "char_end": end_idx,
                    }

            for parent in self.level2parents["sentence"][1:]:
                parent_ids[f"{parent}_sentence_id"] += 1

            row.update(parent_ids)

            self._level_row_cache["sentence"] = row
            current_sent_idx = end_idx
            self.__build_tokens(sentence)

        # sd = [s for s in spacy_doc.sents]
        # print(len(str(sd[0])), sd[0][-1].idx, sd[0][-1].text, sd[1][0].idx, sd[1][0].text, len(sentences[0]))
        # spacy_sentences = [str(s) for s in spacy_doc.sents]
        # if sentences != spacy_sentences:
        #     print(sentences)
        #     print(spacy_sentences)
        #     print(lol)

        # sentences = nltk.sent_tokenize(paragraph)
        # for i, sent in enumerate(sentences):
        #     sent_len = len(sent)

        #     if current_sent_idx == 0:
        #         start = 0
        #     else:
        #         start = doc.find(sent, current_sent_idx) #, current_sent_idx)

        #     end = start + sent_len

        #     row =  {
        #             "id": self.__get_global_id("sentence"),
        #             "paragraph_sentence_id": i,
        #             "text":sent,
        #             "char_start": start,
        #             "char_end": end,
        #             }

        #     for parent in self.level2parents["sentence"][1:]:
        #         parent_ids[f"{parent}_sentence_id"] += 1

        #     row.update(parent_ids)

        #     self._level_row_cache["sentence"] = row

        #     #self.stack_level_data["sentence"].append(row)

        #     current_sent_idx = end #sent_len + 1

        #     self.__build_tokens()


    def __build_paragraphs(self):
        """
        Tokenize text int paragraphs (assumed to be a documents) and creates
        a row/processed unit for each paragraph.  For each sentence call __build_sentences().

        For each par we set a global id, character span and the ids of all parents.

        See __build_tokens() documentation for a more detailed example of format.
        """

        parent_ids = self.__get_structure_ids("paragraph")
        doc, _ = self.__get_parent_text("paragraph")
        current_para_idx = self.__get_char_idx("paragraph")


        paras = self.__paragraph_tokenize(doc)

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
                    "text": para,
                    "char_start": start,
                    "char_end": end,
                    }

            for parent in self.level2parents["paragraph"][1:]:
                parent_ids[f"{parent}_paragraph_id"] += 1

            row.update(parent_ids)

            #self.stack_level_data["paragraph"].append(row)
            self._level_row_cache["paragraph"] = row
            current_para_idx = end

            self.__build_sentences()


    def __build_level_dfs(self, string:str, text_type:str, text_id:str, label:str):
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
        label : str
            label of the given string, if there is any.

        Raises
        ------
        NotImplementedError
            if a text type that is not of the valid ones an error will be thrown. Text type
            has to be either document, paragraph or sentence.
        """
        self._level_row_cache[text_type] = {
                                            "id":text_id,
                                            "text":string
                                            }
        # #self.stack_level_data[text_type].append({
        #                                     "id":text_id,
        #                                     "text":string
        #                                     })

        if text_type == "document":
            self.__build_paragraphs()
        elif text_type == "paragraph":
            self.__build_sentences()
        elif text_type == "sentence":
            self.__build_tokens()
        else:
            raise NotImplementedError(f'"{text_type}" is not a understood type')


    def __prune_hiers(self):
        """
        Given the dataset level we set the hierarchy. For now the heirarchy dicts, found in the Datset Class,
        are containing the full supported hierarchiy; from document to tokens. But if our datset level is for
        example  sentences we need to remove documents and paragraphs for the assumed hierarchy so we know that
        what to process.
        """

        new_level2parents = {self.dataset_level:[]}
        for k, v in self.level2parents.items():

            if k in self.parent2children[self.dataset_level]:
                new_level2parents[k] = v[:v.index(self.dataset_level)+1]

        self.level2parents = new_level2parents


    def __init_setup(self, level:str):
        """
        sets up the stacks where we will add the processed units of each level,
        also set the datset level, e.g. on what level are the samples, and corrects
        the hierarchy thereafter.

        Parameters
        ----------
        level : str
            the level of the unit to be processed, e.g. "sentence", "document", "token" etc
            as this function is called at first processing sample its assumed to be the dataset level.


        """
        self.dataset_level = level

        # storing the current row for each level, used to fetch ids etc for lower lever data
        self._level_row_cache = {}
        self._data_stack = []

        # self.stack_level_data = {level:[]}
        # self.stack_level_data.update({l:[] for l in self.parent2children[level]})
        # self.level_dfs = {l:pd.DataFrame() for l in self.stack_level_data.keys()}

        self.data = pd.DataFrame()
        #self.nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse') #'tokenize,mwt,pos,lemma,depparse'



        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
        #self.nlp.add_pipe("sentencizer", first=True)
        # sent_tok = self.nlp.create_pipe('sentencizer')
        # sent_tok.punct_chars.extend(["\n", "\n\n"]) #, "\n\n"])
        # print(sent_tok.punct_chars)
        # self.nlp.add_pipe(sent_tok, first=True)

        self.__prune_hiers()


    def _clear_stack(self):
        """
        Create dataframes from the list of processed units on each level.
        """
        if self.data.shape == (0,0):
            self.data = pd.DataFrame(self._data_stack)
            self._clean()
        else:
            raise NotImplementedError

        self._data_stack = []
        #del self.nlp
        # for level, stack in self.stack_level_data.items():
        #     self.level_dfs[level] = self.level_dfs[level].append(pd.DataFrame(stack))
        # self.stack_level_data = {l:[] for l in self.stack_level_data.keys()}


    def _clean(self):
        self.data = self.data[~self.data.text.str.contains("\n")]
        self.data = self.data[~self.data.text.str.contains("\t")]
        self.data.reset_index(inplace=True)



    def create_ams(self, method="pre-ac"):

        if method=="pre-ac":
            # self.level_dfs["token"]["am_id"] = np.nan
            # groups = self.level_dfs["token"].groupby("sentence_id")

            self.data["am_id"] = np.nan
            groups = self.data.groupby("sentence_id")

            for sent_id, sent_df in tqdm(groups, desc="labeling tokens with Argumentative Markers"):

                acs = sent_df.groupby("ac_id")
                prev_ac_end = 0
                for ac_id, ac_df in acs:

                    ac_start = min(ac_df["char_start"])
                    ac_end = max(ac_df["char_end"])

                    # more than previous end of ac and less than ac start
                    cond1 = sent_df["char_start"] >= prev_ac_end
                    cond2 = sent_df["char_start"] < ac_start
                    idxs = sent_df[cond1 & cond2].index
                    # self.level_dfs["token"]["am_id"].iloc[idxs] = ac_id
                    self.data["am_id"].iloc[idxs] = ac_id
                    prev_ac_end = ac_end


    def add_samples(self, data:List[dict]):
        """

        A function for preprocessing multiple samples. Simply calls add_sample().


        Parameters
        ----------
        data : List[dict]
            list of dict, where each dict contains the paramaters for add_sample()


        TODO:


        -- multiprocessing
        problem:
        as we are infering global ids from the order of processing
        we cannot just pass the data to multiprocessing and add to
        a shared stacked dict.
        Instead we need to make sure the data storing is NOT shared
        across the processes, then when combining the processed data
        we need to know which split/batch was first and update
        the next split ids appropriately.

        e.g. if we have 2 splits of 1000 samples
        both ids will range from 1-1000, hence for the 2nd split
        we just need to append 1000 to all of the global ids.

        """
        # if not hasattr(self, "dataset_level"):
        #     self.__init_setup(data[0]["text_type"])

        # manager = mp.Manager()
        # stack = manager.dict()
        # pool = mp.Pool(processes=8)
        # for k,v in self.stack_level_data.items():
        #     stack[k] = v
        # self.stack_level_data = stack

        # pool.map(self.add_sample,data)
        # pool.close()
        # pool.join()p

        for sample in tqdm(data, desc="preprocessing data"):
            self.add_sample(**sample, _Preprocessor__single=False)

        self._clear_stack()


    def add_sample(self, text:str, text_type:str, sample_id:int, label=None, __single:bool=True):
        """

        Preprocess a text sample.

        The type of the first text sample given will be assumed to be the dataset_level, i.e.
        the level on which the samples are.

        Parameters
        ----------
        text : str
            the text sample
        text_type : str
            the type of the text sample, e.g. "sentence", "document"
        sample_id : int
            id of the text sample
        label : [type], optional
            if the text sample has a label one can add it, by default None
        __single : bool, optional
            internal paramters for keeping the add_sample fucntion from calling _clear_stack after
            each text sample processed. Useful when processing samples with add_samples(), by default True

        Raises
        ------
        ValueError
            [description]
        """

        text_type = text_type if text_type else self.__infer_text_type(text)

        if not hasattr(self, "dataset_level"):
            self.__init_setup(text_type)

        # if isinstance(text_id,int):
        #     if text_id in self.stack_level_data[text_type]:
        #         raise ValueError("that id already exists")
        # else:
        #     text_id = self.stack_level_data[text_type.shape[0]

        self.__build_level_dfs(text, text_type, sample_id, label)

        if __single:
            self._clear_stack()
