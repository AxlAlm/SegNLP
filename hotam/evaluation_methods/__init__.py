from hotam.evaluation_methods.default import default
from hotam.evaluation_methods.cross_validation import cross_validation

def get_evaluation_method(choice):

    if choice == "default":
        return default
    elif choice == "cross_validation":
        return cross_validation
    else:
        raise NotImplementedError 


__all__ = [
            "get_evaluation_method"
        ]       