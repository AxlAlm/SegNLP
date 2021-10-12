

# basics
import argparse

# am exps
import am_exps

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-model',  type=str, help='select model')
    parser.add_argument('-level',  type=str, help='sample level')
    parser.add_argument('-mode',  type=str, help='Chose if you want to do training or testing, or both')
    parser.add_argument('-gpu',  type=int, help='device id of the GPU you want to use', default = None, required = False)
    args = parser.parse_args()
    
    # fetch the module containing the scripts for the chosen model
    model_script = getattr(am_exps, args.model.lower())

    # call the run function in the model script which will run the experiement
    model_script(
        sample_level = args.level,
        mode = args.mode,
        gpu = args.gpu
        )
