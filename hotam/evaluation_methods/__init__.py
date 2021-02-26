from hotam.evaluation_methods.default import default


def get_evaluation_method(choice):

    if choice == "default":
        return default
    else:
        raise NotImplementedError 


__all__ = [
            "get_evaluation_method"
        ]       