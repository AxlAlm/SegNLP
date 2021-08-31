


# basics
import numpy as np
from typing import Union

#h5py
import h5py


class H5PY_STORAGE:

    def __init__(self, fp: Union[str, h5py.File], name:str, n_dims:int, dtype: np.dtype, fillvalue:int = 0):
        self.name = name
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

        c_n_samples, current_dim1, feature_dim = self.h5py_f[self.name].shape
        dim1, feature_dim = input.shape   
        
        # figure out the new shape
        new_shape = (c_n_samples+1, max(current_dim1, dim1), feature_dim)

        # resize to the new shape
        self.h5py_f[self.name].resize(new_shape)
    
        #add sample
        self.h5py_f[self.name][c_n_samples, :dim1] = input

        
    def close(self):
        self.h5py_f.close()