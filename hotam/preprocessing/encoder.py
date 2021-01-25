
import torch 
import numpy as np
from typing import List
from tqdm import tqdm
import pandas as pd

#axl nlp
from hotam.preprocessing.encoders import *
from hotam import get_logger
from hotam.utils import one_tqdm

logger = get_logger("ENCODER")

class DatasetEncoder:
    """Class for encoding dataset.

    for each given encoding type creates a encoder class and stores it

    padding is supported for all encoders 

    """


    @one_tqdm(desc="Creating Label Encoders")
    def _create_label_encoders(self):
        
        for task in self.all_tasks:
            

            if task == "relation":
                relations = self.task2labels[task]
                _min_rel_value = min(relations)
                _max_rel_value = max(relations)
                max_back_relation = 0 if _min_rel_value > 0 else _min_rel_value
                max_forward_relation = 0 if _max_rel_value < 0 else _max_rel_value
                max_acs = self._get_max_nr_seq("ac")
                self.encoders[task] = RelationEncoder(name=task, max_back_relation=max_back_relation, max_forward_relation=max_forward_relation, max_acs=max_acs)
            else:
                self.encoders[task] = LabelEncoder(name=task, labels=self.task2labels[task])


    @one_tqdm(desc="Encoding Labels")
    def _encode_labels(self):

        for task in self.all_tasks:
            # self.level_dfs["token"][task] = self.level_dfs["token"][task].apply(lambda x: self.encode(x, task))

            if task == "relation":
                
                self.data["_relation"] = self.data["relation"].to_numpy()
                samples = self.data.groupby(self.sample_level+"_id")
                for s_id, df in tqdm(samples, desc="encoding relations"):
                    acs = df.groupby("ac_id")
                    relations = [ac_df["relation"].unique()[0] for ac_id, ac_df in acs]
                    enc_relations = self.encode_list(relations, task)
  
                    for i, (ac_id, ac_df) in enumerate(acs):
                        self.data.loc[ac_df.index,"_relation"] = enc_relations[i]

                self.data["relation"] = self.data.pop("_relation")

            else:
                self.data[task] = self.data[task].apply(lambda x: self.encode(x, task))


    @one_tqdm(desc="Creating Data Encoders")
    def _create_data_encoders(self):
        """encodes the data for each the given encoding types. Also, creates encoding
        models which are saved and can be used at later stages.d
        """
        self.enc2padvalue = {}

        #logger.info("Creating Encoders ...")
        for enc_type in self.encodings:

            if enc_type == "words":
                self.encoders["words"] = WordEncoder()

            elif enc_type == "pos":
                self.encoders["pos"] = PosEncoder()

            elif enc_type == "deprel":
                self.encoders["deprel"] = DepEncoder()

            elif enc_type == "dephead":
                pass

            elif enc_type == "chars":
                self.encoders["chars"] = CharEncoder()

            elif enc_type == "bert_encs":
                self.encoders["bert_encs"] = BertTokEncoder()
            else:
                raise KeyError(f'"{enc_type}" is not a supported encoding')
            
            self.enc2padvalue[enc_type] = 0

    @one_tqdm(desc="Encoding Data")
    def _encode_data(self):
        
        for enc in self.encodings:
            
            if enc == "bert_encs":
                self.__encode_bytepairs("bert_encs")
            if enc == "dephead":
                pass
            else:
                if enc in ["words", "chars"]:
                    self.data[enc] = self.data["text"].apply(lambda x: self.encode(x, enc))
                else:
                    self.data[enc] = self.data[enc].apply(lambda x: self.encode(x, enc))

    
    def __encode_bytepairs(self, enc):


        new_token_rows = []
        desc = "Encoding to BERT byte-pair encodings"
        # for i, token_row in tqdm(self.level_dfs["token"].iterrows(), total=self.level_dfs["token"].shape[0], desc=desc):
        for i, token_row in tqdm(self.data.iterrows(), total=self.data.shape[0], desc=desc):   
            token = token_row["text"]
            enc_ids = self.encode(token, enc)

            if len(enc_ids) > 1:
                
                next_is_I = False
                for enc_id in enc_ids:
                    token_row = token_row.copy()
                    token_row[enc] = enc_id

                    #token_row["bert_text"] = self.decode(enc_id, enc)
                    # if enc_id == enc_ids[-1]:
                    #     token_row["next_is_space"] = True
                    # else:
                    #     token_row["next_is_space"] = False

                    if token_row["seg"] == "O":
                        pass
                    elif token_row["seg"] == "B" and not next_is_I:
                        next_is_I = True
                    else:
                        token_row["seg"] = "I"
                
                    new_token_rows.append(token_row)
            else:
                token_row[enc] = enc_ids[0]
                #token_row["next_is_space"] = True
                #token_row["bert_text"] = token_row["text"]
                new_token_rows.append(token_row)
        

        # self.level_dfs["token"] = pd.DataFrame(new_token_rows)
        self.data = pd.DataFrame(new_token_rows)


    def decode(self, item:int, name:str) -> str:
        """
        decodes an int given a encoder type

        Parameters
        ----------
        item : int
            int to decode
        name : str
            name of encoder

        Returns
        -------
        string
            decoded string
            
        """
        return self.encoders[name].decode(item)


    def encode(self, item:str, name:str) -> int:
        """encodes a string given the encoder type

        Parameters
        ----------
        item : str
            string to encode
        name : str
            name of encoder

        Returns
        -------
        int
            int for encoded string
        """
        return self.encoders[name].encode(item)


    def decode_list(self, item:List[int], name:str) -> List[str]:
        """decodes a list of ints given encoder type

        Parameters
        ----------
        item : List[int]
            items to decode
        name : str
            encoder name
        pad : bool, optional
            if you want to pad or not, by default False

        Returns
        -------
        List[str]
            decoded strings
        """
        return self.encoders[name].decode_list(item)


    def encode_list(self, item:List[str], name:str) -> List[int]:
        """encodes a list of strings given encoder type

        Parameters
        ----------
        item : List[int]
            items to encode
        name : str
            encoder name
        pad : bool, optional
            if you want to pad or not, by default False

        Returns
        -------
        List[int]
            encodes strings
        """
        return self.encoders[name].encode_list(item)
