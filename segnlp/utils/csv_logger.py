

#basics
import os


class CSVLogger:

    def __init__(self, path_to_logs:str, max_lines = 1000) -> None:
        self._path_to_logs = path_to_logs
        self.__create_log_file()
        self._max_lines = 1000
        self._current_lines = 0
        self._current_log_files = len(os.listdir(self._path_to_csv))

    def __create_log_file(self):
        self.log_file = os.path.join(self._path_to_logs, f"log-{self._current_log_files}")
        open(self.log_file, "w")
        self.log_file.close()

    
    def query(self, query="best"):
        pass


    def log_epoch(self, 
                model_id:str,
                random_seed: str,
                epoch_metrics:dict,
                ):
        log_dict = {
                    **{
                        "model_uid":model_id,
                        "random_seed": random_seed,
                        },
                    **{epoch_metrics}
                    }

        if self._current_lines  == self._max_lines:
            self.__create_log_file()

        with open(self.log_file, "a") as f:
            csv_string = ",".join(log_dict.values()) + "\n"
            f.write(csv_string)

        self._current_lines += 1
