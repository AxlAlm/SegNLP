



class ArgumentMiner:


    def __init__(self):
        pass


    def add_model(self,
                model:None,
                #hyperparamate
                tasks:list,
                prediction_level:str,
                sample_level:str, 
                features:list=[],
                encodings:list=[],
                remove_duplicates:bool=False,
                tokens_per_sample:bool=False,
                override:bool=False,
                ):
        """prepares the data
        """

        if prediction_level == "ac" and [t for t in tasks if "seg" in t]:
            raise ValueError("If prediction level is ac you cannot have segmentation as a task")

        feature_levels = set([fm.level for fm in features])
        if "doc" in feature_levels and prediction_level == "token":
            raise ValueError("Having features on doc level is not supported when prediction level is on word level.")
        
        config = {}

        #self.prediction_level = prediction_level
        #self.sample_level = sample_level
        #self.tokens_per_sample = tokens_per_sample
        #self.main_tasks = tasks
        #self.tasks = tasks
        #self.encodings = encodings

        feature2model = {fm.name:fm for fm in features}
        feature_groups = set([fm.group for fm in features])
        feature2dim = {fm.name:fm.feature_dim for fm in features}
        feature2dim.update({
                                group:sum([fm.feature_dim for fm in features if fm.group == group]) 
                                for group in self._feature_groups
                                })
        features ={
                    "models":feature2model,
                    "dims":feature2dim,
                    "groups":feature_groups,
                    "word_features":[fm.name for fm in feature2model.values() if fm.level == "word"]

                    }

        #self.feature2dim["word_embs"] = sum(fm.feature_dim for fm in features if fm.level == "word")
        #self.feature2dim["doc_embs"] = sum(fm.feature_dim for fm in features if fm.level == "doc")
        #self.__word_features = [fm.name for fm in self.feature2model.values() if fm.level == "word"]

        # to later now how we cut some of the padding for each batch
        if self.tokens_per_sample:
            self.__cutmap.update({k:"tok" for k in  encodings + ["word_embs"]})

        #create a hash encoding for the exp config
        #self._exp_hash = self.__create_exp_hash()
        #self._enc_file_name = os.path.join("/tmp/", f"{'-'.join(self.tasks+self.encodings)+self.prediction_level}_enc.json")

        # if remove_duplicates:
        #     self.remove_duplicates()

        # #remove duplicates from splits
        # if remove_duplicates:
        #     if self.duplicate_ids.any():
        #         self.update_splits()


        #if self.prediction_level == "token" or self.tokens_per_sample:
        #   self.max_tok = self._get_nr_tokens(self.sample_level)
        #if self.prediction_level == "ac":
        
        self.max_tok = self._get_nr_tokens(self.sample_level)
        self.max_seq = self._get_max_nr_seq("ac")
        self.max_seq_tok = self._get_nr_tokens(self.prediction_level)
    
        if self.sample_level != "sentence" and ("deprel" in self.encodings or  "dephead" in self.encodings):
            self.max_sent = self._get_max_nr_seq("sentence")
            self.max_sent_tok = self._get_nr_tokens("sentence")
            self.__cutmap["dephead"] = "sent"
            self.__cutmap["deprel"] = "sent"
        

        enc_data_exit = os.path.exists(self._enc_file_name)
        if enc_data_exit and not override:
            self.__load_enc_state()
            self._create_data_encoders()
            self._create_label_encoders()
    
        else:
            self.__fix_tasks()
            self._create_data_encoders()
            self._encode_data() 
            self.__fuse_subtasks()
            self.__get_task2labels()
            self._create_label_encoders()
            self.stats(override=True)
            self._encode_labels()
            self.__save_enc_state()
        
        
        # self.level_dfs["token"].index = self.level_dfs["token"][f"{sample_level}_id"].to_numpy() #.pop(f"{sample_level}_id")
        self.data.index = self.data[f"{sample_level}_id"].to_numpy() #.pop(f"{sample_level}_id")
        
        # self.nr_samples = len(self.level_dfs[self.sample_level].shape[0]
        self.nr_samples = len(self.data[self.sample_level+"_id"].unique())

        if self.sample_level != self.dataset_level:
            self._change_split_level()
        

        self.config = {
                        "prediction_level":prediction_level,
                        "sample_level": sample_level,
                        "tasks": tasks,
                        "subtasks": self.subtasks,
                        "encodings": encodings,
                        "features": self.features,
                        "remove_duplicates": remove_duplicates,
                        "task_labels": self.task2labels,
                        "tracked_sample_ids": {str(s): ids["val"][:20].tolist() for s, ids in self.splits.items()}
                    }

        if hotam.preprocessing.settings["CACHE_SAMPLES"]:
            self.__setup_cache()
        



    def add_dataset(self, dataset):
        pass
