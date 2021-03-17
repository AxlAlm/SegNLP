

#/Users/xalmax/phd/datasets/arg-microtexts/corpus/en

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
        self.data = self.__process_data()


    def name(self):
        return "MTC"