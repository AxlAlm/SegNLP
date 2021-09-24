

import functools


class Memorize:

    def __init__(self, func):
        self.func = func
        self.cache = {}


    def __call__(self, *args, **kwargs):
        
        key = args + tuple(sorted(kwargs.items()))

        if key in self.cache:
            return self.cache[key]
        else:
            value = self.func(*args, **kwargs)
            self.cache[key] = value
            return value


    def __repr__(self):
        return self.func.__doc__


    def __get__(self, obj, objtype):
        # As we self.func will be a callable form another class we need to get the instance
        # of the class self.func is fetched from. __get__ is then called and then we can modify self.func 
        # so its passed with its instansiated object
        return functools.partial(self.__call__, obj)
