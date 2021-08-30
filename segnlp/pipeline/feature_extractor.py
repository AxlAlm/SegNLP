   
# basics
import pandas as pd
import numpy as np


 
class FeatureExtractor:
   

    def _extract_pretrained_features(self, sample:pd.DataFrame):
            
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
                outputs[group_dict["level"]] =  np.concatenate(group_dict["data"], axis=-1), 
            else:
                outputs[group_dict["level"]] = group_name, group_dict["data"][0]

        return outputs
