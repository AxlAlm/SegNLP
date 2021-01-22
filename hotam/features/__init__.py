

def get_feature(feature_name):

    for fm in __all__:
        
        if fm == get_feature or fm == FeatureSet:
            continue
        
        if  feature_name.lower() in fm.name().lower():
            return fm

    raise KeyError(f'"{fm}" is no a supported model"')


from .bow import BOW
from .embeddings import Embeddings
from .document_positions import DocPos
from .dummy import DummyFeature
from .one_hot import OneHots

__all__ = [
            "Embeddings",
            "DocPos",
            "BOW",
            "DummyFeature",
            "OneHots"
        ]