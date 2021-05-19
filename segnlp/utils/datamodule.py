


class PreProcessedDataset(ptl.LightningDataModule):

    def __init__(self, name:str, dir_path:str, prediction_level:str):
        self._name = name
        self._fp = os.path.join(dir_path, f"{name}_data.hdf5")

        with h5py.File(self._fp, "r") as f:
            self._size = f["ids"].shape[0]

        self.prediction_level = prediction_level

        self._stats = pd.read_csv(os.path.join(dir_path, f"{name}_stats.csv"))
        self._stats.columns = ["split_id", "split", "task", "label", "count"]

        with open(os.path.join(dir_path, f"{name}_splits.pkl"), "rb") as f:
            self._splits = pickle.load(f)
        

    def __getitem__(self, key:Union[np.ndarray, list]) -> ModelInput:
        Input = ModelInput()

        with h5py.File(self._fp, "r") as data:
            sorted_key = np.sort(key)
            lengths = data[self.prediction_level]["lengths"][sorted_key]
            lengths_decending = np.argsort(lengths)[::-1]
            Input._ids = data["ids"][key][lengths_decending]
            
            for group in data:

                if group == "ids":
                    continue

                max_len = max(data[group]["lengths"][sorted_key])

                Input[group] = {}
                for k, v in data[group].items():
                    a = v[key]
                
                    if len(a.shape) > 1:
                        a = a[:, :max_len]
                
                    Input[group][k] = a[lengths_decending]
                
            Input.to_tensor()

        return Input
    

    def __len__(self):
        return self._size


    def name(self):
        return self._name

    @property
    def info(self):
        
        structure = { }

        with h5py.File(self._fp, "r") as data:
            for group in data.keys():

                if group == "ids":
                    structure["ids"] = f'dtype={str(data["ids"].dtype)}, shape={data["ids"].shape}'
                    continue

                structure[group] = {}
                for k, v in data[group].items():
                    structure[group][k] = f"dtype={str(v.dtype)}, shape={v.shape}"


        s = f"""
            Structure:        
            
            {json.dumps(structure, indent=4)}

            Size            = {self._size}
            File Size (MBs) = {str(round(os.path.getsize(self._fp) / (1024 * 1024), 3))}
            Filepath       = {self._fp}
            """
        print(s)

    @property
    def stats(self):
        return self._stats

    @property
    def splits(self):
        return self._splits


    def train_dataloader(self):
        # ids are given as a nested list from sampler (e.g [[42, 43]]) hence using lambda x:x[0] to select the inner list.
        sampler = BatchSampler(self.splits[self.split_id]["train"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0], num_workers=segnlp.settings["dl_n_workers"])


    def val_dataloader(self):
        # ids are given as a nested list from sampler (e.g [[42, 43]]) hence using lambda x:x[0] to select the inner list.
        sampler = BatchSampler(self.splits[self.split_id]["val"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0], num_workers=segnlp.settings["dl_n_workers"]) #, shuffle=True)


    def test_dataloader(self):
        sampler = BatchSampler(self.splits[self.split_id]["test"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0], num_workers=segnlp.settings["dl_n_workers"])
