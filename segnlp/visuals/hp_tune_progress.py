
#basics
from tqdm import tqdm
import sys



class HpProgress:

    def __init__(self, hyperparamaters:list, n_seeds:int, show_n:int=3):
        self.hyperparamaters = {i:{"progress":"-", "hp":hp} for i,hp in enumerate(hyperparamaters)}
        self.nr_hps = len(hyperparamaters)
        self.nr_hp_keys = len(hyperparamaters[0].keys())
        self.progress_bar = tqdm(total=self.nr_hps, desc="Hyperparamater Tuning")

        self._header = self._build_row( ["i"] + list(hyperparamaters[0].keys()) + ["prog"] )

        self._ongoing_id = None
        self._ongoing_row = " " * len(self._header)
        self._top_id = None
        self._top_row = " " * len(self._header)

        self._double_underline = u"\u2017" * len(self._header)
        self._singel_underline = u"\u005F"* len(self._header)

        self._show_n = show_n
        self._n_seeds = n_seeds


    def _build_row(self, values):
        double_line = '\u2016'
        row = "{} " + "".join([f"|{{:^6.5}}" for i in range(self.nr_hp_keys)]) + f"{double_line}{{:^8}}"+ double_line
        return row.format(*values)
    

    def _build_tabel(self):

        print(self._header)
        print(self._double_underline)


        print(self._top_row + "<- top")
        print(self._double_underline)

        print(self._ongoing_row + "<- ongoing")
        print(self._double_underline)


        n = 0
        for i, hp in self.hyperparamaters.items():

            if i == self._ongoing_id or i == self._top_id:
                continue

            if hp["progress"] == self._n_seeds:
                p = u'\u2713' #check
            else:
                p = f"{hp['progress']}/4"

            str_values = list(map(str, hp["hp"].values()))
            row_values =  [str(i)] + str_values + [p]
            r = self._build_row(row_values)
            print(r)

            if n >= self._show_n:
                break
            
            print(self._singel_underline)


        if n <= self._show_n:
            print("......")
            print(self._singel_underline)

        sys.stdout.flush()


    def refresh(self, i:int, progress:int):

        self.hyperparamaters[i]["progress"] = progress
        hp = self.hyperparamaters[i]

        if progress == self._n_seeds:
            p = u'\u2713' #check
        else:
            p = f"{hp["progress"]}/4"

        str_values = list(map(str,hp["hp"].values()))
        row_values =  [str(i)] + str_values + [p]
        r = self._build_row(row_values)
        
        self._ongoing_id = i
        self._ongoing_row = r

        self._build_tabel()


    def set_top(i):
    
        hp = self.hyperparamaters[i]

        str_values = list(map(str,hp["hp"].values()))
        row_values =  [str(i)] + str_values + [u'\u2713']
        r = self._build_row(row_values)

        self._top_row = r
        self._top_id  = i

        self._build_tabel()


    def close(self):
        self.progress_bar.close()
    

    def update(self):
        self.progress_bar.update(1)

