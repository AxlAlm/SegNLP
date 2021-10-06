
#basics
from typing import List, Dict, Tuple, Union
from tqdm.auto import tqdm
import pandas as pd
from glob import glob
import os

# pytorch 
import torch

#segnlp
from segnlp import get_logger
import segnlp.utils as utils


class TestLoop:


    def test(self, 
            batch_size : int = 32, 
            ) -> None:

        #loading our preprocessed dataset
        datamodule  = utils.DataModule(
                                path_to_data = self._path_to_data,
                                batch_size = batch_size,
                                cv = 0
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
        model_info_path = os.path.join(self._path_to_hps, best_hp_id + ".json")
        model_info = utils.load_json(model_info_path)

        hyperparamaters = model_info["hyperparamaters"]
        random_seeds = model_info["random_seeds"]

        for random_seed in tqdm(random_seeds, desc= "random_seeds"):

            #model path 
            model_path = glob(os.path.join(self._path_to_models, best_hp_id, f"{random_seeds}*"))[0]
            
            #setup model
            model = self.model(
                            hyperparamaters  = hyperparamaters,
                            label_encoder = self.label_encoder,
                            feature_dims = self.feature2dim,
                            metric = self.metric
                            )
            
            #load model weights
            model.load_state_dict(torch.load(model_path))

            # set model to model to evaluation mode
            model.eval()
            with torch.zero_grad():

                for test_batch in tqdm(test_dataset, desc = "Testing", total = len(test_dataset)):
                    
                    #pass the batch to model
                    model(test_batch)

                    #calculate the metrics
                    metric_container.calc_add(test_batch, "train")


            # Log val epoch metrics
            test_epoch_metrics = model.metrics.calc_epoch_metrics("val")
            self._log_epoch(  
                                epoch = 0,
                                split = "test",
                                hp_id = best_hp_id,
                                random_seed = random_seed,
                                epoch_metrics = test_epoch_metrics,
                                cv = 0
                                )






            # seeds.append(seed_model["random_seed"])

            # with open(seed_model["config_path"], "r") as f:
            #     model_config = json.load(f)

            # hyperparamaters = model_config["args"]["hyperparamaters"]




            # model = deepcopy(self.model)
            # model_config["args"]["label_encoders"] = self.label_encoders
            # model = model.load_from_checkpoint(seed_model["path"], **model_config["args"])
            # scores = trainer.test(
            #                         model=model, 
            #                         test_dataloaders=data_module.test_dataloader(),
            #                         verbose=0
            #                         )

            # test_output = pd.DataFrame(model.outputs["test"])


            # if seg_preds is not None:
            #     test_output["seg"] = "O"

            #     #first we get all the token rows
            #     seg_preds = seg_preds[seg_preds["token_id"].isin(test_output["token_id"])]

            #     # then we sort the seg_preds
            #     seg_preds.index = seg_preds["token_id"]
            #     seg_preds = seg_preds.reindex(test_output["token_id"])

            #     assert np.array_equal(seg_preds.index.to_numpy(), test_output["token_id"].to_numpy())
                
            #     #print(seg_preds["seg"])
            #     test_output["seg"] = seg_preds["seg"].to_numpy()
            #     seg_mask = test_output["seg"] == "O"

            #     task_scores = []
            #     for task in self.config["subtasks"]:
            #         default_none =  "None" if task != "link" else 0
            #         test_output.loc[seg_mask, task] = default_none
            #         task_scores.append(base_metric(
            #                                         targets=test_output[f"T-{task}"].to_numpy(), 
            #                                         preds=test_output[task].to_numpy(), 
            #                                         task=task, 
            #                                         labels=self.config["task_labels"][task]
            #                                         ))

            #   scores = [pd.DataFrame(task_scores).mean().to_dict()]
              


        #     if seed_model["random_seed"] == best_seed:
        #         best_model_scores = pd.DataFrame(scores)
        #         best_model_outputs = pd.DataFrame(test_output)

        #     seed_scores.append(scores[0])
    
        # df = pd.DataFrame(seed_scores, index=seeds)
        # mean = df.mean(axis=0)
        # std = df.std(axis=0)

        # final_df = df.T
        # final_df["mean"] = mean
        # final_df["std"] = std
        # final_df["best"] = best_model_scores.T
        
        # with open(self._path_to_test_score, "w") as f:
        #     json.dump(seed_scores, f, indent=4)
        
        # return final_df, best_model_outputs

