
#basics
from tqdm.auto import tqdm
from glob import glob
import os

# pytorch 
import torch

#segnlp
from segnlp import get_logger
import segnlp.utils as utils


class TestLoop:


    def test(self, 
            monitor_metric : str, 
            batch_size : int = 32, 
            ) -> None:

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

        for random_seed in tqdm(random_seeds, desc= "random_seeds"):

            #model path 
            model_path = glob(os.path.join(self._path_to_models, best_hp_id, f"{random_seeds}*"))[0]
            
            #setup model
            model = self.model(
                        hyperparamaters  = hyperparamaters,
                        label_encoder = self.label_encoder,
                        feature_dims = self.feature2dim,
                        )
            
            #load model weights
            model.load_state_dict(torch.load(model_path))

            # set model to model to evaluation mode
            model.eval()
            with torch.no_grad():

                for test_batch in tqdm(test_dataset, desc = "Testing", total = len(test_dataset), leave = False):
                    
                    #pass the batch to model
                    model(test_batch)

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


        self.rank_test(monitor_metric)
