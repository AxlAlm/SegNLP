
#basics
from tqdm.auto import tqdm
from glob import glob
import os
from typing import Union
from copy import deepcopy
import pandas as pd

# pytorch 
import torch

#segnlp
from segnlp import get_logger
#import segnlp.utils as utils


class TestLoop:


    def test(self, 
            monitor_metric : str, 
            batch_size : int = 32,
            gpus : Union[list, str] = None
            ) -> None:


        device  = "cpu"
        if gpus:      
            gpu = gpus[0] if isinstance(gpus,list) else gpus
            device =  f"cuda:{gpu}"
        

        #loading our preprocessed dataset
        datamodule  = utils.DataModule(
                                        path_to_data = self._path_to_data,
                                        batch_size = batch_size,
                                        label_encoder = self.label_encoder,
                                        cv = 0,
                                        )

        # setting up metric container which takes care of metric calculation, aggregation and storing
        metric_container = utils.MetricContainer(
                                            metric = self.metric,
                                            task_labels = self.task_labels
                                            )

        # create a dataset generator
        test_dataset = datamodule.step("test")

        # we only test the best hp setting
        best_hp_id = self.best_hp

        # load hyperparamaters
        hp_config = utils.load_json(os.path.join(self._path_to_hps, best_hp_id + ".json"))

        hyperparamaters = hp_config["hyperparamaters"]
        random_seeds = hp_config["random_seeds_done"]

        #best_pred_dfs = None
        #top_score = 0
        for random_seed in tqdm(random_seeds, desc= "random_seeds"):

            if "cuda" in device:
                device_id = device[-1]
                #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
                os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID" 
                os.environ['CUDA_VISIBLE_DEVICES'] = device_id
                torch.cuda.set_device(int(device_id))


            #model path 
            model_path = glob(os.path.join(self._path_to_models, best_hp_id, f"{random_seed}*"))[0]
            

            #setup model
            model = self.model(
                        hyperparamaters  = deepcopy(hyperparamaters),
                        label_encoder = self.label_encoder,
                        feature_dims = self.feature2dim,
                        )
            
            #load model weights
            model.load_state_dict(torch.load(model_path))

    
            # move model to specified device
            model = model.to(device)

            # set model to model to evaluation mode
            model.eval()
            #pred_batch_dfs = []
            with torch.no_grad():

                for test_batch in tqdm(test_dataset, desc = "Testing", total = len(test_dataset), leave = False):
                        
                    # set device for batch
                    test_batch.to(device)

                    #pass the batch to model
                    model(test_batch)

                    #add pred_df
                    #pred_batch_dfs.append(test_batch._pred_df)

                    #calculate the metrics
                    metric_container.calc_add(test_batch, "test")


            # Log val epoch metrics
            test_epoch_metrics = metric_container.calc_epoch_metrics("test")
            self._log_epoch(  
                                epoch = 0,
                                split = "test",
                                hp_id = best_hp_id,
                                random_seed = random_seed,
                                epoch_metrics = test_epoch_metrics,
                                cv = 0
                                )

        #     if test_epoch_metrics[monitor_metric] > top_score:
        #         top_score = test_epoch_metrics[monitor_metric]
        #         best_pred_dfs = pred_batch_dfs
        # pred_df = pd.concat(best_pred_dfs)
        # pred_df.to_csv(self._path_to_test_preds)

        self.rank_test(monitor_metric)

        print(" _______________  Mean Scores  _______________")
        mean_df = pd.DataFrame(self._load_logs()[self.best_hp]["test"]).mean().to_frame()
        filter_list = set(["epoch", "hp_id", "random_seed", "cv", "use_target_segs", "freeze_segment_module"])
        index = [c for c in mean_df.index if c not in filter_list]
        mean_df = mean_df.loc[index]

        for baseline in self._baselines:
            mean_df[baseline] = pd.DataFrame(self.baseline_scores()["test"][baseline]).mean()
            
        print(mean_df)
