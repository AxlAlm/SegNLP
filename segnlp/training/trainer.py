# basics
from typing import Callable
from tqdm.auto import tqdm
from collections import Counter
import os

# pytorch
import torch

# segnlp
from segnlp.datasets.base import DataSet
from .train_utils import EarlyStopping
from .train_utils import SaveBest
from .train_utils import ScheduleSampling
from .train_utils import configure_optimizer
from .train_utils import configure_lr_scheduler
from .train_utils import CSVLogger

class Trainer:


    def __init__(self,
                name : str,
                model: torch.nn.Module,
                dataset : DataSet,
                metric_fn : Callable,
                monitor_metric :str, 
                optimizer_config : dict, 
                max_epochs : int,
                path_to_models : str,
                path_to_logs : str,
                patience: int = None,
                lr_scheduler_config : dict = None,
                ground_truth_sampling_k: int = None,
                pretrain_segmenation_k: int = 0,
                overfit_batches_k: int = None,
                device = int,
        ) -> None:

        # init vars
        self.name = name
        self.model = model
        self.dataset = dataset
        self.metric_fn = metric_fn
        self.max_epochs = max_epochs
        self.overfit_batches_k = overfit_batches_k
        self.monitor_metric = monitor_metric
        self.pretrain_segmenation_k = pretrain_segmenation_k
        self.device = device
        
        # init logger
        self.logger = CSVLogger(log_file = os.path.join(path_to_logs, f"{name}.log"))

        # set up a checkpoint class to save best models
        self.checkpointer = SaveBest(model_file = os.path.join(path_to_models, f"{name}.ckpt"))

        # EarlyStopping, if patience is None it will allways return False
        self.early_stopper = EarlyStopping(patience)

        # if we want to use ground truth in segmentation during training we can use
        # the following variable value to based on epoch use ground truth segmentation
        self.target_seg_sampling = ScheduleSampling(
                                                schedule="inverse_sig",
                                                k = ground_truth_sampling_k
                                                )

        # after model is initialized we can setup optimizer
        self.optimizer = configure_optimizer(
                                        model = self.model, 
                                        config = optimizer_config
                                        )

        # setup learning scheduler
        self.lr_scheduler = configure_lr_scheduler(
                                        opt = self.optimizer,
                                        config = lr_scheduler_config
                                        )


        if "cuda" in device:
            device_id = device[-1]
            #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
            os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID" 
            os.environ['CUDA_VISIBLE_DEVICES'] = device_id
            torch.cuda.set_device(int(device_id))



    def epoch(self, split:str,  backwards : bool = True, use_target_segs:bool = False):
            
        sum_metrics = Counter()
        batch_iter = getattr(self.dataset, f"{split}_batches")
        for j, batch in enumerate(batch_iter):
            
            #set device
            batch.to(self.device)
            
            #if we are using sampling
            batch.use_target_segs = use_target_segs

            # pass the batch
            loss = self.model(batch)

            # calc grads
            if backwards:

                # reset the opt grads
                self.optimizer.zero_grad()

                # backward
                loss.backward()

                # clip gradients
                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

                # update paramaters
                self.optimizer.step()


            # calculate the metrics
            sum_metrics += self.metric_fn(batch)
            sum_metrics["loss"] += loss.item()

            # update tqdm
            self.postfix[f"{split}_loss"] = loss.item()
            self.epoch_tqdm.set_postfix(self.postfix)
            self.epoch_tqdm.update(1)


            if j+1 == self.overfit_batches_k:
                break


        avrg_metrics = sum_metrics / (j+1)
      
        return avrg_metrics

    
    def fit(self):

        # keeping track of scores
        self.postfix = { 
                    "train_loss" : 0.0, 
                    "val_loss": 0.0, 
                    f"train_{self.monitor_metric}":0.0,
                    f"val_{self.monitor_metric}":0.0,
                    f"top_val_{self.monitor_metric}": 0.0
                    }
        
        #setup tqmd
        self.epoch_tqdm = tqdm(
                        position = 2, 
                        postfix = self.postfix,
                        leave = False
                        )

        for i in range(self.max_epochs):

            #if we are pretraining or not
            is_pretraining = i < self.pretrain_segmenation_k 
            is_using_target_segs = False if is_pretraining else self.target_seg_sampling(i)

            #freeze modules
            self.model.freeze(freeze_segment_module = is_pretraining)
            
            # training
            self.model.train()
            train_scores = self.epoch("train", is_using_target_segs)
        
            # validation
            self.model.eval()
            with torch.no_grad():
               val_scores  = self.epoch()

            # get the monitor score
            monitor_score = val_scores[self.monitor_metric]

            # if we use a learning scheduler we call step() to make it do its thing 
            # with the optimizer, i.e. change the learning rate in some way
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(monitor_score)

            # if we are pretraining segmentation or using target segmentation we dont update stuff
            if is_pretraining or is_using_target_segs:

                # save best model
                self.checkpointer(self.model, monitor_score)

                #set the top score
                self.postfix[f"top_val_{self.monitor_metric}"] = self.checkpointer._top_score

                # stop if model is not better after n times, set by patience
                if self.early_stopper(monitor_score):
                    break
            
            self.epoch_tqdm.set_postfix(self.postfix)
            self.epoch_tqdm.update(1)


            # log train
            self.logger.log(
                            {**dict(
                                epoch = i,
                                split = "train",
                                is_using_target_segs = is_using_target_segs,
                                is_pretraining = is_pretraining
                                ),
                            **train_scores
                            }
                            )

            # log val
            self.logger.log(
                            {**dict(
                                    epoch = i,
                                    split = "val",
                                    is_using_target_segs = is_using_target_segs,
                                    is_pretraining = is_pretraining
                                    ),
                            **val_scores,
                            }
                            )



    def test(self):
        pass


