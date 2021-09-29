   
# basics
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# h5py
import h5py

#segnlp
from segnlp import get_logger


logger = get_logger(__name__)
 
class PretrainedFeatureExtractor:


    def _init_pretrained_feature_extractor(self, pretrained_features):

        # pretrained featues
        self.feature2model : dict = {fm.name:fm for fm in pretrained_features}
        self.features : list = list(self.feature2model.keys())
        self._feature_groups : set = set([fm.group for fm in pretrained_features])
        self.feature2dim : dict = {fm.name:fm.feature_dim for fm in pretrained_features}
        self.feature2dim.update({
                                group:sum([fm.feature_dim for fm in pretrained_features if fm.group == group]) 
                                for group in self._feature_groups
                                })

        self.feature2param = {f.name:f.params for f in pretrained_features}
        self._use_pwf : bool = "word_embs" in self._feature_groups
        self._use_psf : bool = "seg_embs" in self._feature_groups


    def __extract_sample_features(self, sample:pd.DataFrame):
            
        feature_dict = {}
        sample_length = sample.shape[0]

        for feature, fm in self.feature2model.items():
    
            if fm.level == "doc" and self.prediction_level == "seg":
                
                segs = sample.groupby("seg_id", sort = False)
                feature_matrix = np.zeros((len(segs), fm.feature_dim))
                for i,(seg_id, seg_df) in enumerate(segs):
                    # sent.index = sent["id"]
                    data = sample[sample["seg_id"] == seg_id]

                    if self.argumentative_markers:
                        am = sample[sample["am_id"] == seg_id]
                        data = pd.concat((am,data))

                    #adu.index = adu.pop("seg_id")
                    feature_matrix[i] = fm.extract(data)


            elif fm.level == "word":
                # context is for embeddings such as Bert and Flair where the word embeddings are dependent on the surrounding words
                # so for these types we need to extract the embeddings per context. E.g. if we have a document and want Flair embeddings
                # we first divide the document up in sentences, extract the embeddigns and the put them bsegk into the 
                # ducument shape.
                # Have chosen to not extract flair embeddings with context larger than "sentence".
                if fm.context and fm.context != self.sample_level:

                    contexts = sample.groupby(f"{fm.context}_id", sort = False)

                    sample_embs = []
                    for _, context_data in contexts:
                        sample_embs.extend(fm.extract(context_data)[:context_data.shape[0]])

                    feature_matrix = np.array(sample_embs)
            
                else:
                    #feature_matrix[:sample_length] = fm.extract(sample)[:sample_length]
                    feature_matrix = fm.extract(sample)[:sample_length]

            else:
                feature_matrix = fm.extract(sample)[:sample_length]


            if fm.group not in feature_dict:
                feature_dict[fm.group] = {
                                        "level": "seg" if fm.level == "doc" else "token",
                                        "data":[]
                                        }
            

            feature_dict[fm.group]["data"].append(feature_matrix)


        outputs = {}
        for group_name, group_dict in feature_dict.items():

            if len(group_dict["data"]) > 1:
                outputs[group_name] =  np.concatenate(group_dict["data"], axis=-1)
            else:
                outputs[group_name] = group_dict["data"][0]

        return outputs


    def _preprocess_pretrained_features(self, df : pd.DataFrame) -> None:

        if self._use_pwf:
            #logger.info("Creating h5py file for pretrained word features ... ")

            max_toks = max(df.groupby(level = 0, sort=False).size())
            fdim = self.feature2dim["word_embs"]

            h5py_pwf = h5py.File(self._path_to_pwf, "w")
            h5py_pwf.create_dataset(
                                    "word_embs", 
                                    data = np.random.random((self._n_samples, max_toks, fdim)), 
                                    dtype = np.float64, 
                                    )
    


        if self._use_psf:
            #logger.info("Creating h5py file for pretrained segment features ... ")

            max_segs = max(df.groupby(level = 0, sort=False)["seg_id"].nunique())
            fdim = self.feature2dim["seg_embs"]

            h5py_psf = h5py.File(self._path_to_psf, "w")
            h5py_psf.create_dataset(
                                    "seg_embs", 
                                    data = np.random.random((self._n_samples, max_segs, fdim)), 
                                    dtype = np.float64, 
                                    )
    

        for i, sample in tqdm(df.groupby(level = 0), desc="Preprocessing Pretrained Features"):
            feature_dict = self.__extract_sample_features(sample)

            if self._use_pwf:
                t, _ = feature_dict["word_embs"].shape
                h5py_pwf["word_embs"][i, :t, :] = feature_dict["word_embs"]
            
            if self._use_psf:
                s, _ = feature_dict["seg_embs"].shape
                h5py_psf["seg_embs"][i, :s, :] = feature_dict["seg_embs"]

