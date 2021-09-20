#basic
from glob import glob
from tqdm import tqdm
import pandas as pd
import os
from pathlib import Path
from typing import Tuple, List, Dict
import shutil

#git
from git import Repo

# segnlp
from segnlp import get_logger
import segnlp.utils as u
from segnlp.utils import RangeDict
from segnlp.datasets.base import DataSet




class PE(DataSet):

    """Class for downloading, reading and parsing the Pursuasive Essay dataset found here: 

    https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/argument_annotated_essays/index.en.jsp

    Does not parse the dataset from the available CONLL format as it contain errors, instead its parsed
    directly from the original dataset.

    """

    def __init__(self,
                tasks:list,
                prediction_level:str="token", 
                sample_level:str="document", 
                dump_path:str="/tmp/"
                ):

        task_labels = {
                            "seg": ["O","B","I"],
                            "label":["None", "MajorClaim", "Claim", "Premise"],
                            # Originally stance labels are For and Against for Claims and MajorClaims
                            # and for premsies supports or attacks. 
                            # However, Against and attacks are functional equivalent so will use CON for both
                            # and for For and supports we will use PRO
                            #"stance":["For", "Against", "supports", "attacks"],
                            "link_label": ["support", "attack", "root"],
                            "link": list(range(-11,12,1))
                            }

        self._stance2new_stance = {
                                    "supports": "support", 
                                    "For": "support", 
                                    "Against": "attack", 
                                    "attacks": "attack",
                                    "root": "root"
                                    }
        super().__init__(
                        name="pe",
                        tasks=tasks,
                        prediction_level=prediction_level,
                        sample_level=sample_level,
                        supported_task_labels=task_labels,
                        level="document",
                        supported_tasks=["seg", "label", "link", "link_label"],
                        supported_prediction_levels=["seg", "token"],
                        supported_sample_levels=["document", "paragraph", "sentence"],
                        about="""The corpus consists of argument annotated persuasive essays including annotations of argument components and argumentative relations.""",
                        url="https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/argument_annotated_essays_version_2/index.en.jsp",
                        download_url= "https://github.com/UKPLab/acl2017-neural_end2end_am",
                        dump_path=dump_path,
                        )



    def _download_data(self):

        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)
            Repo.clone_from(self.download_url, self.dump_path)
        
        return os.path.join(self.dump_path, "acl2017-neural_end2end_am")

        # dir = "Paragraph" if self.sample_level == "paragraph" else "Essay"
        # path_to_data = os.path.join(self.dump_path, "data", "conll", dir + "/*.dat.abs.xml")

        # if len(glob.glob(path_to_data)) == 0:
        #     shutil.rmtree(self.dump_path)
        #     self._download_data()
          
        #return path_to_data


    def process_data(self, p):

        path_to_essays = os.path.join(self.dump_path, "data", "conll", "Essay_level" + "/*.dat.abs.xml")
        path_to_paragraphs = os.path.join(self.dump_path, "data", "conll", "Paragraph_Level" + "/*.dat.abs.xml")

        essay_df  = self.__read_conll_data(path_to_essays)
        para_df  = self.__read_conll_data(path_to_paragraphs)

        essay_df["paragraph_id"] = para_df["id"]

        if sample_id == "paragraph":
            

        return essay_df


    def __read_conll_data(self, path_to_data:str):

        files = [os.path.join(path_to_data,f) for f in os.listdir(path_to_data)]
        
        # set_labels = set(['Claim', 'MajorClaim', None, 'Premise'])
        # set_link_labels = set(['Attack', None, 'Support'])
    
        id = 0
        rows = [] 

        for fp in files:
                
            split = fp.rsplit("/",1).split(".")[0]

            with open(fp, "r") as f:
                for line in f:
                    
                    line = line.strip("\n")

                    if not line:
                        id += 1
                        continue
                        
                    token_id, token_str, label = line.split("\t")

                    row = {
                                "id" : id,
                                "token_id": token_id,
                                "str": token_str,
                                "seg": "O",
                                "label": "None",
                                "link": "None",
                                "link_label": "root",
                                "split" : split
                                }
        
                    
                    if label == "O":
                        rows.append(row)
                        continue
                                    
                    IOB, complex_label = label.split("-")
                    row["seg"] = IOB

                    labels = complex_label.split(":")

                    if labels[0] == "MajorClaim":
                        row["label"] = labels[0]

                    elif labels[0] == "Claim":
                        row["label"] =  labels[0]
                        row["link_label"] =labels[1]

                    else:
                        row["label"] =  labels[0]
                        row["link"] = labels[1]
                        row["link_label"] = labels[2]

                    rows.append(row)

    
        
        return  pd.DataFrame(rows)