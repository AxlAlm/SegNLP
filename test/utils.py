
# basic
import os
import pandas as pd

# segnlp
from segnlp.data import Sample
from segnlp.utils import RangeDict
from segnlp.data import Batch

class Utils:

    def create_sample(self):
        doc = "This is a paragrap which is also a segments. \nThis is a not a segment. However, here starts a segement. This, right after this comma, is also a segment."
        return Sample(doc)


    def create_and_label_sample(self):

        sample = self.create_sample()

        span2label = RangeDict({
            (0, 44): {'span_id': 0, 'seg_id': 0, 'label': 'A', 'link': 0, 'link_label': 'root'}, 
            (45, 70): {'span_id': 1, 'seg_id': -1, 'label': 'None', 'link': -1, 'link_label': 'None'}, 
            (71, 104): {'span_id': 2, 'seg_id': 1, 'label': 'A', 'link': 2, 'link_label': 'A'},
            (105, 133): {'span_id': 3, 'seg_id': -1, 'label': 'None', 'link': -1, 'link_label': 'None'}, 
            (134, 152): {'span_id': 4, 'seg_id': 2, 'label': 'B', 'link': 2, 'link_label': 'root'}, 
        })
        task_labels = {
                        "seg": ["O", "B", "I"],
                        "seg+label": ["O", "B_A", "I_A", "B_B", "I_B"],
                        "label": ["None", "A", "B"],
                        "link_label": ["root", "A", "B"],
                        "link": []
                        }
        sample.add_span_labels(
                                    span2label, 
                                    task_labels = task_labels, 
                                    label_ams = True
                                    )
        return sample
    

    def load_gold_df(self):
        path_to_gold_sample = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gold_sample.csv")
        return pd.read_csv(path_to_gold_sample, index_col = 0)
    

    def create_batch(self, size:int = 10):
        return Batch([self.create_and_label_sample() for  i in range(size)])
