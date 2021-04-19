

from .bow import BOW
from .embeddings import GloveEmbeddings, FlairEmbeddings, BertEmbeddings
from .unit_position import UnitPos
from .dummy import DummyFeature
from .one_hot import OneHots


def get_feature(feature_name):

    if  feature_name.lower() == "glove":
        return GloveEmbeddings

    elif  feature_name.lower() == "flair":
        return FlairEmbeddings

    elif  feature_name.lower() == "bert":
        return BertEmbeddings
    
    elif feature_name.lower() == "bow":
        return BOW

    elif feature_name.lower() == "unitpos":
        return UnitPos

    raise KeyError(f'"{feature_name}" is no a supported model"')


__all__ = [
            "get_feature",
            "GloveEmbeddings",
            "FlairEmbeddings",
            "BertEmbeddings",
            "UnitPos",
            "BOW",
            "DummyFeature",
            "OneHots"
        ]