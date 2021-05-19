    


class CoreLoops:


    def _fit(    self,
                hyperparamaters:dict,
                ptl_trn_args:dict={},
                save_choice:str = "best",  
                random_seed:int = 42,
                monitor_metric:str = "val_f1",
                model_id:str=None
                ):


        if model_id is None:
            model_id = create_uid("".join(list(map(str, hyperparamaters.keys())) + list(map(str, hyperparamaters.values()))))

        set_random_seed(random_seed)

        hyperparamaters["random_seed"] = random_seed
        self.dataset.batch_size = hyperparamaters["batch_size"]
        hyperparamaters["monitor_metric"] = monitor_metric


        model = deepcopy(self.model)
    
        if self.exp_logger:
            ptl_trn_args["logger"] = self.exp_logger
        else:
            ptl_trn_args["logger"] = None
        
        mid_folder = "top" if not self.__hp_tuning else "tmp"
        exp_model_path = os.path.join(self._path_to_models, "tmp", model_id, str(random_seed))
        
        if os.path.exists(exp_model_path):
            shutil.rmtree(exp_model_path)
            
        os.makedirs(exp_model_path, exist_ok=True) 

        model_args = self.__get_model_args(
                                            model=model, 
                                            hyperparamaters=hyperparamaters, 
                                            )

        self.__save_model_config(
                                model_args=model_args,
                                save_choice=save_choice,
                                monitor_metric=monitor_metric,
                                exp_model_path=exp_model_path
                                )


        if self.exp_logger:
            self.exp_logger.set_id(model_id)
            self.exp_logger.log_hyperparams(hyperparamaters)

            if isinstance(exp_logger, CometLogger):
                self.exp_logger.experiment.add_tags([self.project, self.id])
                self.exp_logger.experiment.log_others(exp_config)


        ptl_trn_args = get_ptl_trainer_args( 
                                        ptl_trn_args=ptl_trn_args,
                                        hyperparamaters=hyperparamaters, 
                                        exp_model_path=exp_model_path,
                                        save_choice=save_choice, 
                                        #prefix=model_id,
                                        )

        model_fp, model_score = get_evaluation_method(self.evaluation_method)(
                                                                                model_args = model_args,
                                                                                ptl_trn_args = ptl_trn_args,
                                                                                dataset = self.dataset,
                                                                                save_choice=save_choice,
                                                                                )

        return {
                "model_id":model_id, 
                "score":model_score, 
                "monitor_metric":monitor_metric,
                "random_seed":random_seed,
                "path":model_fp, 
                "config_path": os.path.join(exp_model_path, "model_config.json")
                }
 

    def train(self,
                    hyperparamaters:dict,
                    ptl_trn_args:dict={},
                    n_random_seeds:int=6,
                    random_seed:int=None,
                    save_choice:str="best",
                    monitor_metric:str = "val_f1",
                    ss_test:str="aso",
                    debug:bool=False,
                    override:bool=False
                    ):

        # if ptl_trn_args.get("gradient_clip_val", 0.0) != 0.0:
        #     hyperparamaters["gradient_clip_val"] = ptl_trn_args["gradient_clip_val"]
        
        self.__hp_tuning = True
        keys = list(hyperparamaters.keys())
        set_hyperparamaters = self.__create_hyperparam_sets(hyperparamaters)

        # if we have done previous tuning we will start from where we ended, i.e. 
        # from the previouos best Hyperparamaters
        if os.path.exists(self._path_to_model_info):

            with open(self._path_to_model_info, "r") as f:
                best_model_info = json.load(f)

            best_scores = best_model_info["scores"]
        else:
            best_scores = None
            best_model_info = {}

        
        if os.path.exists(self._path_to_hp_hist):
            with open(self._path_to_hp_hist, "r") as f:
                hp_hist = json.load(f)
        else:
            hp_hist = {}
    
        create_hp_uid = lambda x: create_uid("".join(list(map(str, x.keys()))+ list(map(str, x.values()))))
        hp_dicts = {create_hp_uid(hp):{"hyperparamaters":hp} for hp in set_hyperparamaters}
        hp_dicts.update(hp_hist)

        hpp = HpProgress(
                        keys=keys,
                        hyperparamaters=hp_dicts,
                        n_seeds = n_random_seeds,
                        best=best_model_info.get("uid", None),
                        show_n=3,
                        debug=debug,
                        )

        for hp_uid, hp_dict in hp_dicts.items():    
            
            if hp_uid in hp_hist and not override:
                logger.info(f"Following hyperparamaters {hp_uid} has already been tested over n seed. Will skip this set.")
                continue

            best_model_score = None
            best_model = None
            model_scores = []
            model_outputs = []

            if random_seed is not None and isinstance(random_seed, int):
                random_seeds = [random_seed]
            else:
                random_seeds = random_ints(n_random_seeds)

            for ri, random_seed in enumerate(random_seeds, start=1):
                hpp.refresh(hp_uid, 
                            progress = ri, 
                            best_score = best_model_score if best_model_score  else "-",
                            mean_score = np.mean(model_scores) if model_scores else "-"
                            )

                output = self._fit(
                                    hyperparamaters=hp_dict["hyperparamaters"],
                                    ptl_trn_args = ptl_trn_args,
                                    save_choice=save_choice,
                                    random_seed=random_seed,
                                    monitor_metric=monitor_metric,
                                    model_id=hp_uid,
                                    )

                score = output["score"]

                if best_model_score is None:
                    best_model_score = score
                    best_model = output
                else:
                    if score > best_model_score:
                        best_model_score = score
                        best_model = output
                
                model_outputs.append(output)
                model_scores.append(score)


            hp_dicts[hp_uid]["uid"] = hp_uid
            hp_dicts[hp_uid]["scores"] = model_scores
            hp_dicts[hp_uid]["score_mean"] = np.mean(model_scores)
            hp_dicts[hp_uid]["score_median"] = np.median(model_scores)
            hp_dicts[hp_uid]["score_max"] = np.max(model_scores)
            hp_dicts[hp_uid]["score_min"] = np.min(model_scores)
            hp_dicts[hp_uid]["monitor_metric"] = monitor_metric
            hp_dicts[hp_uid]["std"] = np.std(model_scores)
            hp_dicts[hp_uid]["ss_test"] = ss_test
            hp_dicts[hp_uid]["n_random_seeds"] = n_random_seeds
            hp_dicts[hp_uid]["hyperparamaters"] = hp_dict["hyperparamaters"]
            hp_dicts[hp_uid]["outputs"] = model_outputs
            hp_dicts[hp_uid]["best_model"] = best_model
            hp_dicts[hp_uid]["best_model_score"] = best_model_score
            hp_dicts[hp_uid]["progress"] = n_random_seeds


            if best_scores is not None:
                is_better, p, v = self.__model_comparison(model_scores, best_scores, ss_test=ss_test)
                hp_dicts[hp_uid]["p"] = p
                hp_dicts[hp_uid]["v"] = v

            if best_scores is None or is_better:
                best_scores = model_scores

                hp_dicts[hp_uid]["best_model"]["path"] = hp_dicts[hp_uid]["best_model"]["path"].replace("tmp","top")
                hp_dicts[hp_uid]["best_model"]["config_path"] = hp_dicts[hp_uid]["best_model"]["config_path"].replace("tmp","top")

                updated_outputs = []
                for d in hp_dicts[hp_uid]["outputs"]:
                    d["path"] = d["path"].replace("tmp","top")
                    d["config_path"] = d["config_path"].replace("tmp","top")
                    updated_outputs.append(d)
                
                hp_dicts[hp_uid]["outputs"] = updated_outputs
                best_model_info = hp_dicts[hp_uid]

                if os.path.exists(self._path_to_top_models):
                    shutil.rmtree(self._path_to_top_models)
                    
                shutil.move(self._path_to_tmp_models, self._path_to_top_models)

                hpp.set_top(hp_uid)

            if os.path.exists(self._path_to_tmp_models):
                shutil.rmtree(self._path_to_tmp_models)

            hpp.update()
        
        hpp.close()
            
        with open(self._path_to_hp_hist, "w") as f:
            json.dump(hp_dicts, f, indent=4)

        with open(self._path_to_model_info, "w") as f:
            json.dump(best_model_info, f, indent=4)

        return best_model_info


    def test(   self, 
                model_folder:str=None,
                ptl_trn_args:dict={},
                monitor_metric:str = "val_f1",
                seg_preds:str=None,
                ):

        self.dataset.split_id = 0


        with open(self._path_to_model_info, "r") as f:
            model_info = json.load(f)

        models_to_test =  model_info["outputs"]
        best_seed = model_info["best_model"]["random_seed"]
        

        best_model_scores = None
        best_model_outputs = None
        seed_scores = []
        seeds = []

        for seed_model in models_to_test:

            seeds.append(seed_model["random_seed"])

            with open(seed_model["config_path"], "r") as f:
                model_config = json.load(f)

            hyperparamaters = model_config["args"]["hyperparamaters"]
            self.dataset.batch_size = hyperparamaters["batch_size"]

            trainer = setup_ptl_trainer( 
                                        ptl_trn_args=ptl_trn_args,
                                        hyperparamaters=hyperparamaters, 
                                        exp_model_path=None,
                                        save_choice=None, 
                                        )

            model_config["args"]["model"] = deepcopy(self.model)
            model_config["args"]["label_encoders"] = self.__pp_encoders
            model = PTLBase.load_from_checkpoint(seed_model["path"], **model_config["args"])
            scores = trainer.test(
                                    model=model, 
                                    test_dataloaders=self.dataset.test_dataloader(),
                                    verbose=0
                                    )

            test_output = pd.DataFrame(model.outputs["test"])


            if seg_preds is not None:
                test_output["seg"] = "O"

                #first we get all the token rows
                seg_preds = seg_preds[seg_preds["token_id"].isin(test_output["token_id"])]

                # then we sort the seg_preds
                seg_preds.index = seg_preds["token_id"]
                seg_preds = seg_preds.reindex(test_output["token_id"])

                assert np.array_equal(seg_preds.index.to_numpy(), test_output["token_id"].to_numpy())
                
                #print(seg_preds["seg"])
                test_output["seg"] = seg_preds["seg"].to_numpy()
                seg_mask = test_output["seg"] == "O"

                task_scores = []
                for task in self.config["subtasks"]:
                    default_none =  "None" if task != "link" else 0
                    test_output.loc[seg_mask, task] = default_none
                    task_scores.append(base_metric(
                                                    targets=test_output[f"T-{task}"].to_numpy(), 
                                                    preds=test_output[task].to_numpy(), 
                                                    task=task, 
                                                    labels=self.config["task2labels"][task]
                                                    ))

                scores = [pd.DataFrame(task_scores).mean().to_dict()]
              


            if seed_model["random_seed"] == best_seed:
                best_model_scores = pd.DataFrame(scores)
                best_model_outputs = pd.DataFrame(test_output)

            seed_scores.append(scores[0])
    
        df = pd.DataFrame(seed_scores, index=seeds)
        mean = df.mean(axis=0)
        std = df.std(axis=0)

        final_df = df.T
        final_df["mean"] = mean
        final_df["std"] = std
        final_df["best"] = best_model_scores.T
        
        with open(self._path_to_test_score, "w") as f:
            json.dump(seed_scores, f, indent=4)
        
        print(final_df)
        return final_df, best_model_outputs

