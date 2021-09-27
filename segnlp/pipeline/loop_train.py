

#basics
from segnlp.utils.array import ensure_numpy
from tqdm.auto import tqdm
from typing import Union
import os
from time import time

#pytroch
import torch

#segnlp
from segnlp import utils
from segnlp.utils import datamodule




class TrainLoop:


    def __train_loop(self, 
                    model_id:str,
                    hyperparamaters: dict,
                    monitor_metric: str,               
                    cv : int = 0,
                    device : Union[str, torch.device] = "cpu",
                    overfit_n_batches : int = None
                    ):


        # some hyperparamters we need to configure training
        max_epochs = hyperparamaters["general"]["max_epochs"]
        gradient_clip_val = hyperparamaters["general"].get("gradient_clip_val", None)
        batch_size = hyperparamaters["general"]["batch_size"]
        patience = hyperparamaters["general"].get("patience", None)
        use_target_segs_k = hyperparamaters["general"].get("use_target_segs_k", None)
        #token_module_freeze_k = hyperparamaters["general"].get("token_module_freeze", False)
        freeze_segment_module_k = hyperparamaters["general"].get("freeze_segment_module_k", False)            

        #loading our preprocessed dataset
        datamodule  = utils.DataModule(
                                path_to_data = self._path_to_data,
                                batch_size = batch_size,
                                label_encoder = self.label_encoder,
                                cv = cv,
                                )

        logger = utils.CSVLogger(
                                path_to_logs = self._path_to_logs, 
                                model_id = model_id
                                )


        # setting up metric container which takes care of metric calculation, aggregation and storing
        metric_container = utils.MetricContainer(
                                            metric = self.metric,
                                            task_labels = self.task_labels,
                                            )

        # set up a checkpoint class to save best models
        path_to_model = os.path.join(self._path_to_models, model_id + ".ckpt")
        checkpointer = utils.SaveBest(path_to_model = path_to_model)

        # EarlyStopping, if patience is None it will allways return False
        early_stopper = utils.EarlyStopping(patience)


        # if we want to use ground truth in segmentation during training we can use
        # the following variable value to based on epoch use ground truth segmentation
        target_seg_sampling = utils.ScheduleSampling(
                                            schedule="inverse_sig",
                                            k=use_target_segs_k
                                            )
                                            
        # set up model
        model = self.model(
                        hyperparamaters  = hyperparamaters,
                        label_encoder = self.label_encoder,
                        feature_dims = self.feature2dim,
                        metric = self.metric,                    
                        )
        

        # move model to specified device
        model = model.to(device)


        # after model is initialized we can setup out optimizers and learning schedulers
        optimizer, lr_scheduler  = utils.configure_optimizers(
                                        model = model, 
                                        hyperparamaters = hyperparamaters
                                        )

        # train batches
        postfix = { 
                    "train_loss" : 0.0, 
                    "val_loss": 0.0, 
                    f"train_{monitor_metric}":0.0,
                    f"val_{monitor_metric}":0.0,
                    f"top_val_{monitor_metric}": 0.0
                    }
        
        #setup tqmd
        epoch_tqdm = tqdm(
                        position=2, 
                        postfix = postfix
                        )

        for epoch in range(max_epochs):

            # we make sure to generate our batches each epoch so that we can shuffle 
            # the whole dataset so we dont keep feeding the model the same batches each epoch
            # NOTE! the dataset är generators so nothing is re-loaded, processed or anything
            train_dataset = datamodule.step(split = "train")
            val_dataset = datamodule.step(split = "val")
 
            #setup tqmd
            epoch_tqdm.reset(total = len(train_dataset) + len(val_dataset))
            epoch_tqdm.set_description(desc = f"Epoch {epoch}")


            # will also freeze and set modules to skip if needed
            cond1 = epoch < freeze_segment_module_k
            cond2 = freeze_segment_module_k == -1
            freeze_segment_module = (cond1 or cond2) and freeze_segment_module_k != False


            # We can can help tasks which are dependent on the segmentation to be feed
            # ground truth segmentaions instead of the predicted segmentation during traingin
            use_target_segs = False
            if not freeze_segment_module and use_target_segs_k:
                use_target_segs = target_seg_sampling(epoch)

            #freeze modules
            model.freeze(freeze_segment_module = freeze_segment_module,)

            # Sets the model to training mode.
            model.train()
            for ti, train_batch in enumerate(train_dataset):
                
                #set device
                train_batch.to(device)
                
                #if we are using sampling
                train_batch.use_target_segs = use_target_segs

                # pass the batch
                loss = model(train_batch)

                #calculate the metrics
                metric_container.calc_add(train_batch, "train")
                
                # reset the opt grads
                optimizer.zero_grad()

                # calc grads
                loss.backward()

                #clip gradients
                if gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)

                # update paramaters
                optimizer.step()

                postfix["train_loss"] = loss.item()
                epoch_tqdm.set_postfix(postfix)
                epoch_tqdm.update(1)


                if overfit_n_batches and ti+1 >= overfit_n_batches:
                    break


            # Validation Loop
            total_val_loss = 0
            model.eval()
            with torch.no_grad(set_to_none=True):
                for vi, val_batch in enumerate(val_dataset):

                    #set device
                    val_batch.to(device)
                                            
                    # pass the batch
                    val_loss = model(
                                val_batch, 
                                )
                    total_val_loss += val_loss.item()
                
                    metric_container.calc_add(val_batch, "val")

                    postfix["val_loss"] = val_loss.item()
                    epoch_tqdm.set_postfix(postfix)
                    epoch_tqdm.update(1)

                    if overfit_n_batches and vi+1 >= overfit_n_batches:
                        break

            # Log train epoch metrics
            train_epoch_metrics = metric_container.calc_epoch_metrics("train")
            logger.log_epoch(  
                                epoch = epoch,
                                split = "train",
                                model_id = model_id,
                                epoch_metrics = train_epoch_metrics,
                                cv = cv
                                )

            postfix[f"train_{monitor_metric}"] = train_epoch_metrics[monitor_metric]


            # Log val epoch metrics
            val_epoch_metrics = metric_container.calc_epoch_metrics("val")
            logger.log_epoch(  
                                epoch = epoch,
                                split = "val",
                                model_id = model_id,
                                epoch_metrics = val_epoch_metrics,
                                cv = cv
                                )
            postfix[f"val_{monitor_metric}"] = val_epoch_metrics[monitor_metric]


            # fetch the score we will decide saving and early stoping on
            score = val_epoch_metrics[monitor_metric]

            # save model
            checkpointer(model, score)

            #set the top score
            postfix[f"top_val_{monitor_metric}"] = checkpointer._top_score

            # stop if model is not better after n times, set by patience
            if early_stopper(score):
                break
            
            # if we use a learning scheduler we call step() to make it do its thing 
            # with the optimizer, i.e. change the learning rate in some way
            if lr_scheduler is not None:
                lr_scheduler.step(total_val_loss)

            epoch_tqdm.set_postfix(postfix)
            epoch_tqdm.update(1)
        


    def __cv_loop(self,
                model_id:str,
                hyperparamaters: dict,
                monitor_metric: str,
                device : Union[str, torch.device] = "cpu"
                ):

        for i in range(n_cvs):
           self.__train_loop(
                        model_id = model_id + f"cv={i}",
                        hyperparamaters = hyperparamaters,
                        monitor_metric = monitor_metric,
                        cv = 1,
                        device = device,
                    )            


    def fit(self,
            model_id:str,
            hyperparamaters: dict,
            monitor_metric: str, 
            device: Union[str, torch.device] = "cpu",
            overfit_n_batches : int = None
            ) -> None:

    
        if self.evaluation_method == "cv":
            self.__cv_loop(
                            model_id = model_id,
                            hyperparamaters = hyperparamaters,
                            monitor_metric = monitor_metric,
                            device = device,
                            overfit_n_batches = overfit_n_batches
                        )
        else:
            self.__train_loop(
                        model_id = model_id,
                        hyperparamaters = hyperparamaters,
                        monitor_metric = monitor_metric,
                        device = device,
                        overfit_n_batches = overfit_n_batches,
                    )