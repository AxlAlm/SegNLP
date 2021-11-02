
#basic
import functools
import numpy as np
from time import time
import _pickle as pkl
import json
from typing import Callable, List, Dict, Tuple
from datetime import datetime
import pytz
from tqdm.auto import tqdm
import os
import sys
import requests
import random
import zipfile
import hashlib
from glob import glob
from pathlib import Path
import re
import pandas as pd
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import random

#torch
import torch

#segnlp
from segnlp import get_logger


logger = get_logger("MISC-UTILS")



def check_file(path):
    
    if os.path.exists(path):
        return path


def check_gpu(self, gpu:int, verbose=1) -> Tuple[bool, torch.device]:
    """checks if there is a gpu device available

    Parameters
    ----------
    gpu : int
        gpu to check for 
    verbose : int, optional
        print information or not, by default 1

    Returns
    -------
    Tuple[bool, torch.device]
        returns a bool for if the gpu is available or not and the device
    """
    gpu_available = torch.cuda.is_available()
    device = torch.device(f'cuda:{gpu}' if gpu_available and isinstance(gpu,int) else 'cpu')

    if verbose != 0:
        logger.info("----------------------")
        logger.info(f"nr GPUs found: {torch.cuda.device_count()}")

        if "cuda" in str(device):
            logger.info(f"Will use GPU number: {gpu}")
        else:
            logger.info("will use CPU")
        logger.info("----------------------")

    return gpu_available, device


class RangeDict(dict):
    def __getitem__(self,query_item):
        if type(query_item) == int:
            for k in self:
                if query_item >= k[0] and query_item <= k[1]:
                    return self[k]
            raise KeyError(query_item)
        else:
            return super().__getitem__(query_item)

    def get(self, query_item, default=None):
        try:
            return self.__getitem__(query_item)
        except KeyError:
            return default


def pickle_data(data, file_path):
    with open(file_path, "wb") as f:
        pkl.dump(data,f,  protocol=4)


def load_pickle_data(file_path):
    with open(file_path, "rb") as f:
        data = pkl.load(f)
    return data

def save_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f,  indent=4)


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def timer(func):

	@functools.wraps(func)
	def calc_time(*args, **kwargs):

		start = time()
		output = func(*args, **kwargs)
		end = time()
		logger.info("Time taken to run {}: {}".format(func, end-start))
		return output

	return calc_time


def one_tqdm(desc:str):
    def decorator(f):
        def wrapper(*args, **kwargs):
            pbar = tqdm(total=1, desc=desc)
            # try:
            f(*args, **kwargs)
            # except error as e:
            #     return e
            pbar.update(1)
            pbar.close()
        
        return wrapper
    
    return decorator


def get_time():
    sweden = pytz.timezone('Europe/Stockholm')
    time = datetime.now().astimezone(sweden)
    #timestamp = timestamp.timestamp()
    return time



def download(url:str, save_path:str, desc:str = ""):

    if os.path.exists(save_path):
        return None

    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length')) #response.iter_content(chunk_size=max(int(total/1000), 1024*1024)), 
    progress_bar = tqdm(total=total, unit='iB', unit_scale=True, desc=desc)

    if total is None:
        raise RuntimeError("Empty Data")

    with open(save_path, 'wb') as f:
        for data in response.iter_content(chunk_size=1024):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

    if total != 0 and progress_bar.n != total:
        raise RuntimeError("ERROR, something went wrong")


def unzip(zip_path:str, save_path:str):
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(save_path)


def dynamic_update(src, v, pad_value=0): 

    a = np.array(list(src.shape[1:]))
    b = np.array(list(v.shape))
    new_shape = np.maximum(a, b)
    new_src = np.full((src.shape[0]+1, *new_shape), pad_value, dtype=src.dtype)

    if len(v.shape) > 2:
        new_src[:src.shape[0],:src.shape[1], :src.shape[2]] = src
        new_src[src.shape[0],:v.shape[0], :v.shape[1]] = v
    #if len(v.shape) == 2:
    #    new_src[:src.shape[0],:src.shape[1], :] = src
    #    new_src[src.shape[0],:v.shape[0], :] = v
    else:
        new_src[:src.shape[0],:src.shape[1]] = src
        new_src[src.shape[0],:v.shape[0]] = v

    return new_src


def set_random_seed(seed, using_gpu:bool=False):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_uid(string):
    uid = str(hashlib.sha256(string.encode('utf-8')).hexdigest())
    return uid


def random_ints(n):
    #create a list of n random seed independent of the random seed set. Uses the timestamp
    ts = int(get_time().timestamp())
    rs = RandomState(MT19937(SeedSequence(ts)))
    return rs.randint(10**6,size=(n,)).tolist()


def get_device(module):
    return next(module.parameters()).device





def freeze_module(module : torch.nn.Module):
    #freeze all of the paramaters
    for name, param in module.named_parameters():
        param.requires_grad = False


def write_data(fp:str, data:str, mode:str):
    with open(fp, mode) as f:
        f.write(data)



def read_file(fp:str, line_fn : Callable = None):
    with open(fp, "r") as f:
        return [l if line_fn is None else line_fn(l) for l in f.read().split("\n") if l.strip()]

    