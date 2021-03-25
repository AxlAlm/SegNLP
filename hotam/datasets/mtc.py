

#/Users/xalmax/phd/datasets/arg-microtexts/corpus/en
import xml.etree.ElementTree as ET
import glob

class MTC(DataSet):

    def __init__(self, path_to_data:str, dump_path="/tmp/"):
        #super().__init__()
        self.dump_path = dump_path
        #self._dataset_path = "datasets/pe/data"
        #self._splits = self.__splits()

        # also called Edge Types
        self.argumentative_functions = ["support", "attack", "linked", "central claim", "normal", "example", "rebut", "undercut"]
        
        #also called ADU Types
        self.labels = ["pro", "opp"]

        self._2new_stance = {
                                    "supports":"PRO", 
                                    "Against":"CON", 
                                    "For":"PRO", 
                                    "attacks":"CON",
                                    }
        #self._tasks = ["ac", "relation", "stance"]
        self._tasks = ["label", "link", "link_label"]
        self.__task_labels = {
                            "label":["MajorClaim", "Claim", "Premise"],
                            "link_label": self.argumentative_functions,
                            "link": set()
                            }

        self.level = "document"
        self.about = """The arg-microtexts corpus features 112 short argumentative texts. All texts were originally written in German and have been professionally translated to English. """
        self.url = "https://github.com/peldszus/arg-microtexts"
        self.data = self.__process_data(path_to_data)


    def name(self):
        return "MTC"


    def __process_data(self, path_to_data):
        xml_files = glob.glob(path_to_data + "/*.xml")

        for f in xml_files:
            tree = ET.parse(f)
            root = tree.getroot()



#         <?xml version='1.0' encoding='UTF-8'?>
# <arggraph id="micro_b001" topic_id="waste_separation" stance="pro">
#   <edu id="e1"><![CDATA[Yes, it's annoying and cumbersome to separate your rubbish properly all the time.]]></edu>
#   <edu id="e2"><![CDATA[Three different bin bags stink away in the kitchen and have to be sorted into different wheelie bins.]]></edu>
#   <edu id="e3"><![CDATA[But still Germany produces way too much rubbish]]></edu>
#   <edu id="e4"><![CDATA[and too many resources are lost when what actually should be separated and recycled is burnt.]]></edu>
#   <edu id="e5"><![CDATA[We Berliners should take the chance and become pioneers in waste separation!]]></edu>
#   <adu id="a1" type="opp"/>
#   <adu id="a2" type="opp"/>
#   <adu id="a3" type="pro"/>
#   <adu id="a4" type="pro"/>
#   <adu id="a5" type="pro"/>
#   <edge id="c6" src="e1" trg="a1" type="seg"/>
#   <edge id="c7" src="e2" trg="a2" type="seg"/>
#   <edge id="c8" src="e3" trg="a3" type="seg"/>
#   <edge id="c9" src="e4" trg="a4" type="seg"/>
#   <edge id="c10" src="e5" trg="a5" type="seg"/>
#   <edge id="c1" src="a1" trg="a5" type="reb"/>
#   <edge id="c2" src="a2" trg="a1" type="sup"/>
#   <edge id="c3" src="a3" trg="c1" type="und"/>
#   <edge id="c4" src="a4" trg="c3" type="add"/>
# </arggraph>
