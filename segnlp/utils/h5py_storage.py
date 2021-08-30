


# basics
import numpy as np
from typing import Union

#h5py
import h5py


class H5PY_STORAGE:

    def __init__(self, fp: Union[str, h5py.File], name:str, n_dims:int, dtype: np.dtype, fillvalue:int = 0):
        self.h5py_f = h5py.File(fp, "w")
        self.h5py_f.create_dataset(
                                    name, 
                                    data = np.zeros(tuple([0]*n_dims)), 
                                    dtype = dtype, 
                                    chunks = True, 
                                    maxshape = tuple([None for v in range(n_dims)]), 
                                    fillvalue = fillvalue
                                    )

    def append(self, input: np.ndarray):
        
        i = self.h5py_f.shape[0]
        n_dims = len(input.shape)
        nr_rows = self.h5py_f.shape[0] + input.shape[0]
        max_shape = np.maximum(self.h5py_f.shape[1:], input.shape[1:])
        new_shape = (nr_rows, *max_shape)
        
        self.h5py_f.resize(new_shape)

        if n_dims == 2:
            self.h5py_f[i:,:input.shape[1]] = input

        elif n_dims > 2:
            self.h5py_f[i:,:input.shape[1], :input.shape[2]] = input

        else:
            raise NotImplementedError()

        
    def close(self):
        self.h5py_f.close()