
#basics
from tqdm import tqdm
import sys
import os

                            #best_score= "{:.2%}".format(best_model_score) if best_model_score  else "-",
                            #mean_score= "{:.2%}".format(np.mean(model_scores)) if model_scores else "-"

class HpProgress:

    def __init__(self, 
                keys:int, 
                hyperparamaters:list,
                n_seeds:int, 
                best:str,
                show_n:int=4, 
                debug:bool=False
                ):
        self.hyperparamaters = hyperparamaters
        self.keys = keys
        self.nr_hps = len(hyperparamaters)
        self.nr_hp_keys = len(keys)

        self._header = self._build_row(keys + ["uid", "prog", "best", "mean"] )
        self._width = len(self._header)

        self._ongoing_id = None
        self._ongoing_row = " " * self._width 
        self._top_id = None
        self._top_row = " " * self._width 

        self._double_underline = u"\u2017" * self._width 
        self._singel_underline = u"\u005F"* self._width 

        self._show_n = show_n
        self._n_seeds = n_seeds
        self._debug = debug

        if best:
            self.set_top(best)
        else:
            self._build_tabel()
        self.progress_bar = tqdm(total=self.nr_hps, desc="Hyperparamater Tuning")



    def _build_row(self, values):
        double_line = u'\u2016'
        row = "".join([f"|{{:^6.5}}" for i in range(self.nr_hp_keys)]) + double_line + f"{double_line}{{:^8}}"*4 + double_line
        return row.format(*values)
    

    def _build_tabel(self, last=False):
        if not self._debug:
            os.system('cls' if os.name == 'nt' else 'clear')
        print()

        name = "Hyperparamater Tuning"
        side = "-" * int(((self._width/2) - (len(name)/2)))
        banner = side + "Hyperparamater Tuning" + side
        print(banner)

        print(self._header)
        print(self._double_underline)


        print(self._top_row + "<- top" + " \u2B50")
        print(self._double_underline)

        print(self._ongoing_row + "<- ongoing")
        print(self._singel_underline)


        n = 0
        for uid, hp in self.hyperparamaters.items():

            progress = hp.get('progress',"-")

            if uid == self._ongoing_id or uid == self._top_id:
                continue

            if progress == self._n_seeds:
                p = u'\u2713' #check
            else:
                p = f"{progress}/{self._n_seeds}"

            best_model_score = hp.get("best_model_score","-")
            best_model_score = "{:.2%}".format(best_model_score) if best_model_score != "-" else "-"
            mean_score = hp.get("mean_score","-")
            mean_score = "{:.2%}".format(mean_score) if mean_score != "-" else "-"

        
            str_values = [str(v) for k,v in hp["hyperparamaters"].items() if k in self.keys]
            row_values = str_values + [uid, p, best_model_score, mean_score]
            r = self._build_row(row_values)
            print(r)

            if n >= self._show_n:
                break
            
            print(self._singel_underline)


        if n <= self._show_n and self.nr_hps > self._show_n:
            print("......")
            print(self._singel_underline)

        print()
        if hasattr(self, "progress_bar") and not last:
            self.progress_bar.refresh()

        if not last:
            print()
            print((side*2) + "Training Model" + (side*2))
            

    def refresh(self, uid:int, progress:int, best_score:float, mean_score:float):

        self.hyperparamaters[uid]["progress"] = progress
        self.hyperparamaters[uid]["best_score"] = best_score
        self.hyperparamaters[uid]["mean_score"] =  mean_score
        hp = self.hyperparamaters[uid]

        p = f"{hp['progress']}/{self._n_seeds}"

        best_model_score = "{:.2%}".format(best_score) if best_score != "-" else "-"
        mean_score = "{:.2%}".format(mean_score) if mean_score != "-" else "-"

        str_values = [str(v) for k,v in hp["hyperparamaters"].items() if k in self.keys]
        row_values =  str_values + [uid, p, best_model_score, mean_score]
        r = self._build_row(row_values)
        
        self._ongoing_id = uid
        self._ongoing_row = r

        self._build_tabel()


    def set_top(self, uid):
            
        hp = self.hyperparamaters[uid]

        best_model_score = hp.get("best_model_score","-")
        best_model_score = "{:.2%}".format(best_model_score) if best_model_score != "-" else "-"
        mean_score = hp.get("mean_score","-")
        mean_score = "{:.2%}".format(mean_score) if mean_score != "-" else "-"

        str_values = [str(v) for k,v in hp["hyperparamaters"].items() if k in self.keys]
        row_values = str_values + [uid, u'\u2713', best_model_score, mean_score]
        r = self._build_row(row_values)

        self._top_row = r
        self._top_id  = uid

        self._build_tabel()


    def close(self):
        self.progress_bar.close()
        self._ongoing_id = " " * self._width 
        self._ongoing_row = " " * self._width 
        self._build_tabel(last=True)
    

    def update(self):
        self.progress_bar.update(1)

