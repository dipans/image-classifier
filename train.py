from time import time, sleep
from process_input_args import get_training_input_args
from model_factory import get_model_instance

def main():
    
    start_time = time()
    in_arg = get_training_input_args()
    
    print("Command Line Arguments:",
              "\nData Directory =", in_arg.data_dir,
              "\nArch =", in_arg.arch, 
              "\nLearning Rate =", in_arg.learning_rate,
              "\nHidden Units =", in_arg.hidden_units,
              "\nGPU Enabled =", in_arg.gpu, 
              "\nEpochs =", in_arg.epochs,
              "\nsave directory =", in_arg.save_dir)

    model = get_model_instance(in_arg)
    model.train()
    
    end_time = time()
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )

if __name__ == "__main__":
    main()
