

#basics
import os


class CSVLogger:

    def __init__(self, path_to_logs:str, model_id:str, max_lines = 1000 ) -> None:
        self._path_to_logs : str = path_to_logs
        self._model_id : str = model_id
        self._max_lines : int = 1000
        self._current_lines : int = 0
        self._first : bool = True


    def __check_create_log_file(self):
        
        n_log_files = len(os.listdir(self._path_to_logs))

        if self._current_lines == self._max_lines or self._first:
            self.log_file = os.path.join(self._path_to_logs, f"{self._model_id}-{n_log_files}.log")

            # create the file
            open(self.log_file, "w")

            self._first = False
            return True

    
    def query(self, query="best"):
        raise NotImplementedError


    def log_epoch(self, 
                epoch : int,
                split : str,
                model_id : str,
                epoch_metrics : dict,
                cv : int
                ):                            

        # create dict
        log_dict = {
                    "epoch": epoch,
                    "split": split,
                    "model_id":model_id,
                    #"random_seed": random_seed,
                    "cv":cv
                    }
        log_dict.update(epoch_metrics)

        #setup file
        new_file = self.__check_create_log_file()

        with open(self.log_file, "a") as f:
            
            # create a header if file is new
            if new_file:
                f.write(",".join(list(log_dict.keys())) + "\n")
            
            # row
            f.write(",".join([str(v) for v in log_dict.values()]) + "\n")

        self._current_lines += 1
