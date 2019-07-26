import torch
import os
import json
from time import time, sleep
from process_input_args import get_predict_input_args


def instantiate_new_model(arch_class_name):
    """
    This function responsible to instantiate appropriate classifier using the
    checkpoint file name. On completion of training, checkpoint file is saved
    using the class name. this method with the help of reflection instantiate
    new instance of the type of the checkpoint file name.
    
    Using the object of same class that is used to train, allows us to apply 
    the same configuration was used for training. This eliminates the need
    of explicitly redefine the classifier again.

    Parameters:
        arch_class_name - This is the class name determined from the checkpoint
                        file name
    Return:
        classifier - Classifer object will the optional attributes set to it.
    """
    module = __import__('model_factory')
    flower_classifer_class = getattr(module, arch_class_name)
    flower_classifer = flower_classifer_class('', 'test')
    return flower_classifer

def load_checkpoint(filepath, gpu=False):
    """
    This function responsible to load the checkpointed model details and instantiate
    new classifier object with the help of the filename.
    
    Based on the argument gpu model is loaded in cpu or cuda

    Parameters:
        filepath - This is path of the checkpoint 
    Return:
        classifier - Classifer object will the optional attributes set to it.
    """
    if gpu==True:
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    classifier = instantiate_new_model(filepath.split('/')[-1].split('.')[0])
    classifier.hidden_units = checkpoint['hidden_input_size']
    classifier.define_classifier()
    classifier.model.class_to_idx = checkpoint['class_to_idx']
    classifier.model.load_state_dict(checkpoint['state_dict'])
    classifier.model.to(device)
    classifier.gpu = gpu
    return classifier

def validate_args(in_arg):
    """
    Run validation on the argument for path to image and checkpoint directory.
    
    Parameters:
        in_arg - parsed command line arguments
    Return:
        None
    """
    if not os.path.isfile(in_arg.path_to_image):
        raise Exception('Invalid image file: {}'.format(in_arg.path_to_image))
        
    if not os.path.isdir(in_arg.checkpoints):
        raise Exception('Invalid checkpoint directory: {}'.format(in_arg.checkpoints))

def process_prediction(in_arg, classifier_name):
    """
    Function drives the prediction process making appropriate helper
    function call. Once the prediction is done, function prints the 
    result as a summary.
    
    Parameters:
        in_arg - parsed command line arguments
        classifier_name - This is the class name determined from the checkpoint
                        file name
    Return:
        None
    """
    classifier = load_checkpoint(in_arg.checkpoints + '/' + classifier_name, in_arg.gpu)
    probs, classes = classifier.predict(in_arg.path_to_image, in_arg.top_k)   
    if in_arg.cat_name != None:
        with open(in_arg.cat_name, 'r') as f:
            if in_arg.gpu == True:
                classes = classes.cpu()
            cat_to_name = json.load(f)
            idx_to_class = dict((idx, cls) for cls, idx in classifier.model.class_to_idx.items())
            pred_classes = classes.detach().numpy()[0]
            pred_labels = [cat_to_name[idx_to_class[x]] for x in pred_classes]

    #Display result
    probs = probs.detach().numpy()[0].tolist()
    classes = classes.detach().numpy()[0].tolist()
    print('\nPrediction summary using model: {}'.format(classifier_name.split('_')[0]),
          '\n=======================================================')
    if in_arg.cat_name != None:
        print('{}\t{}\t{}\n'.format('Class Index', 'Probablity', 'Class Name'))
    else:
        print('{}\t{}\n'.format('Class Index', 'Probablity'))
    for i in range(in_arg.top_k):
        if in_arg.cat_name != None:
            print('{}\t\t{:0.2f}%\t\t{}'.format(classes[i], probs[i]*100, pred_labels[i]))
        else:
            print('{}\t\t{:0.2f}%'.format(classes[i], probs[i]*100))
    
            
def main():
    
    start_time = time()
    
    in_arg = get_predict_input_args()
    print("Command Line Arguments:",
              "\nPath to image =", in_arg.path_to_image,
              "\nPath to checkpoints =", in_arg.checkpoints,
              "\nK top classes =", in_arg.top_k, 
              "\nJSON file for cat name =", in_arg.cat_name,
              "\nGPU Enabled =", in_arg.gpu)
    
    validate_args(in_arg)

    items = os.listdir(in_arg.checkpoints)
    for item in items:
        if '.pth' in item or '.pt' in item: 
            process_prediction(in_arg, item)
    
    
    end_time = time()
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )

if __name__ == "__main__":
    main()