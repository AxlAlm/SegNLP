
# basics
from typing import Union
import numpy as np
import pandas as np


# matplotlib
import matplotlib.pyplot as plt


class Visuals:


    def plot(self, 
            key : Union[list, str], 
            split : Union[list, str],
            ):
        pass

        # df = self._load_logs()
        # df.set_index(["hp_id", "random_seed", "split"], inplace = True)

        # for hp_id in self.hp_ids:

        #     df.loc[(hp_id, split)]


    

    def plot_hp_id(self, hp_id: Union[str,int], split : str, metric:str, mean : bool = False):
    
        df = self._load_logs()
        df.set_index(["hp_id", "random_seed", "split"], inplace = True)

        if isinstance(hp_id, int):
            hp_id = self.hp_ids[int]

        hp_info = self._get_hps[hp_id]
        random_seeds = hp_info["random_seeds"]
        
        fig, ax = plt.subplots()
        ax.set_title(f"{metric} for {hp_id}")
        ax.set_ylabel(metric)
        ax.set_xlabel('Epoch')
        
        ys = []
        for i, random_seed in enumerate(random_seeds, start = 1):
            
            #select data
            seed_df = df.loc[(hp_id, random_seed, split), [metric, "epoch"]]
            
            #sort just to make sure epoch are in order
            seed_df.sort_values("epoch", inplace = True)
            
            #select values
            y = seed_df[metric].to_numpy() / i
            x = seed_df["epoch"].to_numpy()
            
            if not mean:
                ax.plot(x, y)
            
            ys.append(y)
        

        ys = np.vstack(ys)
        
        if mean:
            ax.plot(x, np.mean(ys, axis = 0))
        
        ax.fill_between(x, np.minimum(*ys), np.maximum(*ys), alpha=0.1)
        plt.show()

