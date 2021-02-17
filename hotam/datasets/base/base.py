


class DataSet:


    def __init__(self, batch_size:int):
        pass






    def train_dataloader(self):
        sampler = BatchSampler(self.splits[self.split_id]["train"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0]) #, shuffle=True)


    def val_dataloader(self):
        sampler = BatchSampler(self.splits[self.split_id]["val"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0]) #, shuffle=True)


    def test_dataloader(self):
        sampler = BatchSampler(self.splits[self.split_id]["test"], batch_size=self.batch_size, drop_last=False)
        return DataLoader(self, sampler=sampler, collate_fn=lambda x:x[0])
