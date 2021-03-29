#/Users/xalmax/phd/datasets/arg-microtexts/corpus/en
import xml.etree.ElementTree as ET
import glob
import re
import os
import pandas as pd
import numpy as np

#hotam
from hotam.utils import RangeDict
from hotam.datasets.base import DataSet

#sklearn
from sklearn.model_selection import KFold

#git
from git import Repo

#<?xml version='1.0' encoding='UTF-8'?>
# <arggraph id="micro_b001" topic_id="waste_separation" stance="pro">
#   <edu id="1"><![CDATA[Yes, it's annoying and cumbersome to separate your rubbish properly all the time.]]></edu>
#   <edu id="2"><![CDATA[Three different bin bags stink away in the kitchen and have to be sorted into different wheelie bins.]]></edu>
#   <edu id="3"><![CDATA[But still Germany produces way too much rubbish]]></edu>
#   <edu id="4"><![CDATA[and too many resources are lost when what actually should be separated and recycled is burnt.]]></edu>
#   <edu id="5"><![CDATA[We Berliners should take the chance and become pioneers in waste separation!]]></edu>
#   <adu id="1" type="opp"/>
#   <adu id="2" type="opp"/>
#   <adu id="3" type="pro"/>
#   <adu id="4" type="pro"/>
#   <adu id="5" type="pro"/>

#   <edge id="c6" src="1" trg="1" type="seg"/>
#   <edge id="c7" src="2" trg="2" type="seg"/>
#   <edge id="c8" src="3" trg="3" type="seg"/>
#   <edge id="c9" src="4" trg="4" type="seg"/>
#   <edge id="c10" src="5" trg="5" type="seg"/>

#   <edge id="c1" src="1" trg="5" type="reb"/>
#   <edge id="c2" src="2" trg="1" type="sup"/>
#   <edge id="c3" src="3" trg="c1" type="und"/>
#   <edge id="c4" src="4" trg="c3" type="add"/>
# </arggraph>


class MTC(DataSet):

    def __init__(self, 
                tasks:list,
                prediction_level:str="token", 
                sample_level:str="paragraph", 
                dump_path="/tmp/mtc"
                ):


        self.level = "paragraph"
        self.prediction_level=prediction_level, 
        self.sample_level=sample_level, 
        self.input_level= self.level
        self.dump_path = dump_path
        self._tasks = ["seg", "label", "link", "link_label"]
        self._task_labels = {
                            "label": ["pro", "opp"],
                            "link_label": ["None", "sup", "exa", "add", "reb", "und"],
                            "link": set()
                            }

        self.about = """The arg-microtexts corpus features 112 short argumentative texts. All texts were originally written in German and have been professionally translated to English. """
        self.url = "https://github.com/peldszus/arg-microtexts"

        assert prediction_level in ["token", "unit"]
        assert sample_level in ["paragraph", "sentence"]
        
        unpacked_tasks = set()
        for task in tasks:
            for st in tasks.split("+"):
                if st in unpacked_tasks:
                    raise RuntimeError(f"{st} found in more than one task")
                else:
                    st.add(st)
                    
        assert set(subtasks).issubset(set([self._tasks]))


        self._download_path = self.__download_data()
        self.data = self.__process_data()
        self._splits = self.__splits()


    def name(self):
        return "MTC"


    def __download_data(self):
        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)
            Repo.clone_from(self.url, self.dump_path)

        return os.path.join(self.dump_path,"corpus", "en")


    def __splits(self):
        ### MTC normaly uses Cross Validation
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        ids = np.arange(len(self))
        splits = {i:{"train": train_index,  "test":test_index} for i, (train_index, test_index) in enumerate(kf.split(ids))}
        return splits


    def __process_data(self):
        
        def sort_info(x):
            letter = re.findall(r"(?<=micro_)\w", x.rsplit("/",1)[1])[0]
            nr = int(re.findall(r"(?!0)\d+", x.rsplit("/",1)[1])[0])
            return letter, nr

        xml_files = sorted(glob.glob(self._download_path + "/*.xml"), key=sort_info)
        text_files = sorted(glob.glob(self._download_path + "/*.txt"), key=sort_info)
            
        data = []
        for i in range(len(xml_files)):
            # WE assume that each number in each ID is refering to an unique EDU.
            # for each "id" for any tag type EXCEPT (tag==edge and tag.attrib["type"] == seg),
            # we remove the letter, making all ids numbers. e.g. a1 -> 1, e1 -> 1, c1 -> 1
            # This is bacause the number always refers to a particular EDU EXCEPT in the above mentioned case where 
            # any EDGE ID c{i>max(EDU_ID)} are all of type SEG and only map ei to ai.             
            #
            # follow test will not break
            #
            # if c.tag == "edge":
            #   if c.attrib["type"] == "seg":
            #      assert c.attrib["src"][1:], c.attrib["trg"][1:]
            #   assert c.attrib["id"][1:], c.attrib["src"][1:]
            
            file_id = int(re.findall(r"(?!0)\d+", xml_files[i])[0])
                
            edus = {}
            
            
            with open(text_files[i]) as f:
                text = f.read()
                
            tree = ET.parse(xml_files[i])
            root = tree.getroot()
            
            start = 0
            for i,c in enumerate(root):
                
                if c.tag == "edu":
                    ID = c.attrib["id"][1:]
                    text_len = len(c.text)
                    edus[ID] = {
                                        "label": None, 
                                        "link":int(ID)-1, 
                                        "link_label":None,
                                        "span_id": int(ID)-1,
                                        "unit_id": None,
                                        "span": (start, start+text_len)
                                        }
                    
                    start += (text_len+1)
                
                elif c.tag == "adu":
                    ID = c.attrib["id"][1:]
                    edus[ID]["label"] = c.attrib["type"]
                    edus[ID]["unit_id"] = int(ID)-1
                                        
                elif c.tag == "edge":
                    link_label = c.attrib["type"]
                    
                    if c.attrib["type"] == "seg":
                        assert c.attrib["src"][1:], c.attrib["trg"][1:]
                        continue
                    
                    assert c.attrib["id"][1:], c.attrib["src"][1:]
                
                    ID = c.attrib["id"][1:]
                    edus[ID]["link_label"] = c.attrib["type"]

                    link = (int(c.attrib["trg"][1:])-1) - (int(ID)-1)
                    edus[ID]["link"] = link

                    self._task_labels["link"].add(link)
                
                
            assert len(text) == start-1, f"text={len(text)},  start={start}"
            
            span_labels = RangeDict({e.pop("span"):e for _,e in edus.items()})

            data.append({
                        "sample_id":file_id,
                        "text":text, 
                        "text_type":"document",
                        "span_labels": span_labels
                        })


        self._task_labels["link"] = sorted(self._task_labels["link"])
        return pd.DataFrame(data)