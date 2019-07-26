from flower_classifier import Flower_Classifier
from torchvision import models
from torch import nn, optim
from collections import OrderedDict
from workspace_utils import active_session
import torch

#This is the available model archs for flower classification, any addition
#will needed and alteration of this array
supported_arch = ['alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18',
                  'resnet34', 'resnet50', 'resnet101', 'resnet152',
                  'densenet121', 'densenet161', 'densenet169', 'densenet201']

def set_optional_attr(in_args, classifier):
    """
    This function applies all the optional commandline arguments. This is designed to
    call after the instantiation of classifier object

    Parameters:
        in_args - This is parsed command line arguments
        classifier - This is an object of type Flower_Classifier. All the optional
                   attributes of this object will be set using setters.
    Return:
        classifier - Classifer object will the optional attributes set to it.
    """
    if in_args.save_dir != None:
        classifier.save_dir = in_args.save_dir
    if in_args.learning_rate != None:
        classifier.learning_rate = in_args.learning_rate
    if in_args.hidden_units != None:
        classifier.hidden_units = in_args.hidden_units
    if in_args.epochs != None:
        classifier.epochs = in_args.epochs
    if in_args.gpu != None:
        classifier.gpu = in_args.gpu
    return classifier


def instantiate_new_model(in_args):
    """
    This function responsible to instantiate appropriate classifier based on the
    model architecture.

    Parameters:
        in_args - This is parsed command line arguments
    Return:
        classifier - Classifer object will the optional attributes set to it.
    """
    if in_args.arch=='alexnet':
        return set_optional_attr(in_args, Alexnet_Classifier(in_args.data_dir, Flower_Classifier.EXEC_MODES[0]))
    elif in_args.arch=='vgg11':
        return set_optional_attr(in_args, VGG11_Classifier(in_args.data_dir, Flower_Classifier.EXEC_MODES[0]))
    elif in_args.arch=='vgg13':
         return set_optional_attr(in_args, VGG13_Classifier(in_args.data_dir, Flower_Classifier.EXEC_MODES[0]))
    elif in_args.arch=='vgg16':
         return set_optional_attr(in_args, VGG16_Classifier(in_args.data_dir, Flower_Classifier.EXEC_MODES[0]))
    elif in_args.arch=='vgg19':
         return set_optional_attr(in_args, VGG19_Classifier(in_args.data_dir, Flower_Classifier.EXEC_MODES[0]))
    elif in_args.arch=='resnet18':
        return set_optional_attr(in_args, Resnet18_Classifier(in_args.data_dir, Flower_Classifier.EXEC_MODES[0]))
    elif in_args.arch=='resnet34':
        return set_optional_attr(in_args, Resnet34_Classifier(in_args.data_dir, Flower_Classifier.EXEC_MODES[0]))
    elif in_args.arch=='resnet50':
        return set_optional_attr(in_args, Resnet50_Classifier(in_args.data_dir, Flower_Classifier.EXEC_MODES[0]))
    elif in_args.arch=='resnet101':
        return set_optional_attr(in_args, Resnet101_Classifier(in_args.data_dir, Flower_Classifier.EXEC_MODES[0]))
    elif in_args.arch=='resnet152':
        return set_optional_attr(in_args, Resnet152_Classifier(in_args.data_dir, Flower_Classifier.EXEC_MODES[0]))
    elif in_args.arch=='densenet121':
        return set_optional_attr(in_args, Densenet121_Classifier(in_args.data_dir, Flower_Classifier.EXEC_MODES[0]))
    elif in_args.arch=='densenet161':
        return set_optional_attr(in_args, Densenet161_Classifier(in_args.data_dir, Flower_Classifier.EXEC_MODES[0]))
    elif in_args.arch=='densenet169':
        return set_optional_attr(in_args, Densenet169_Classifier(in_args.data_dir, Flower_Classifier.EXEC_MODES[0]))
    elif in_args.arch=='densenet201':
        return set_optional_attr(in_args, Densenet201_Classifier(in_args.data_dir, Flower_Classifier.EXEC_MODES[0]))
    else:
        raise Exception('Model is not supported')


def get_model_instance(in_args):
    """
    This function responsible to instantiate appropriate classifier based on the
    model architecture by calling instantiate_new_model function. It also 
    validates across the validity of the data directory and if arch is supported
    by this implementation

    Parameters:
        in_args - This is parsed command line arguments
    Return:
        classifier - Classifer object will the optional attributes set to it.
    """
    if in_args.data_dir == None:
        raise Exception('A valid data directory must be provided')

    if in_args.arch not in supported_arch:
        raise Exception('Model is not supported')
    else:
        return instantiate_new_model(in_args)
    
class Alexnet_Classifier(Flower_Classifier):
    """
    This is a concrete implementation of Flower_Classifier to use alexnet 
    """
    def __init__(self, *args, **kwargs):
        super(Alexnet_Classifier, self).__init__(*args, **kwargs)

    def _set_model(self):
        self._model = models.alexnet(pretrained=True)
    
    def define_classifier(self):
        self._model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(self._input_size, self._hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dp1', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(self._hidden_units, self._hidden_units)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(self._hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    def _set_input_size(self):
        self._input_size = 9216

class VGG11_Classifier(Flower_Classifier):
    """
    This is a concrete implementation of Flower_Classifier to use vgg11 
    """
    def __init__(self, *args, **kwargs):
        super(VGG11_Classifier, self).__init__(*args, **kwargs)

    def _set_model(self):
        self._model = models.vgg11(pretrained=True)
    
    def _set_input_size(self):
        self._input_size = 25088

class VGG13_Classifier(Flower_Classifier):
    """
    This is a concrete implementation of Flower_Classifier to use vgg13 
    """
    def __init__(self, *args, **kwargs):
        super(VGG13_Classifier, self).__init__(*args, **kwargs)

    def _set_model(self):
        self._model = models.vgg13(pretrained=True)
    
    def _set_input_size(self):
        self._input_size = 25088

class VGG16_Classifier(Flower_Classifier):
    """
    This is a concrete implementation of Flower_Classifier to use vgg16 
    """
    def __init__(self, *args, **kwargs):
        super(VGG16_Classifier, self).__init__(*args, **kwargs)

    def _set_model(self):
        self._model = models.vgg16(pretrained=True)
        
    def _set_input_size(self):
        self._input_size = 25088

class VGG19_Classifier(Flower_Classifier):
    """
    This is a concrete implementation of Flower_Classifier to use vgg19 
    """
    def __init__(self, *args, **kwargs):
        super(VGG19_Classifier, self).__init__(*args, **kwargs)

    def _set_model(self):
        self._model = models.vgg19(pretrained=True)
        
    def _set_input_size(self):
        self._input_size = 25088

class Resnet18_Classifier(Flower_Classifier):
    """
    This is a concrete implementation of Flower_Classifier to use resnet18 
    """
    def __init__(self, *args, **kwargs):
        super(Resnet18_Classifier, self).__init__(*args, **kwargs)

    def _set_model(self):
        self._model = models.resnet18(pretrained=True)
        
    def _set_input_size(self):
        self._input_size = 512
        
    def define_classifier(self):
        self._model.fc = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(self._input_size, self._hi9216dden_units)),
                          ('relu1', nn.ReLU()),
                          ('dp1', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(self._hidden_units, self._hidden_units)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(self._hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
    def _set_optimizer(self):
        self._optimizer = optim.Adam(self._model.fc.parameters(), lr=self._learning_rate)

class Resnet34_Classifier(Flower_Classifier):
    """
    This is a concrete implementation of Flower_Classifier to use resnet34
    """
    def __init__(self, *args, **kwargs):
        super(Resnet34_Classifier, self).__init__(*args, **kwargs)

    def _set_model(self):
        self._model = models.resnet34(pretrained=True)
        
    def _set_input_size(self):
        self._input_size = 512
        
    def define_classifier(self):
        self._model.fc = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(self._input_size, self._hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dp1', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(self._hidden_units, self._hidden_units)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(self._hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
    def _set_optimizer(self):
        self._optimizer = optim.Adam(self._model.fc.parameters(), lr=self._learning_rate)

class Resnet50_Classifier(Flower_Classifier):
    """
    This is a concrete implementation of Flower_Classifier to use resnet50 
    """
    def __init__(self, *args, **kwargs):
        super(Resnet50_Classifier, self).__init__(*args, **kwargs)

    def _set_model(self):
        self._model = models.resnet50(pretrained=True)
        
    def _set_input_size(self):
        self._input_size = 2048
        
    def define_classifier(self):
        self._model.fc = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(self._input_size, self._hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dp1', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(self._hidden_units, self._hidden_units)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(self._hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
    def _set_optimizer(self):
        self._optimizer = optim.Adam(self._model.fc.parameters(), lr=self._learning_rate)
      
class Resnet101_Classifier(Flower_Classifier):
    """
    This is a concrete implementation of Flower_Classifier to use resnet101 
    """
    def __init__(self, *args, **kwargs):
        super(Resnet101_Classifier, self).__init__(*args, **kwargs)

    def _set_model(self):
        self._model = models.resnet101(pretrained=True)
        
    def _set_input_size(self):
        self._input_size = 2048
        
    def define_classifier(self):
        self._model.fc = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(self._input_size, self._hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dp1', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(self._hidden_units, self._hidden_units)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(self._hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
    def _set_optimizer(self):
        self._optimizer = optim.Adam(self._model.fc.parameters(), lr=self._learning_rate)

class Resnet152_Classifier(Flower_Classifier):
    """
    This is a concrete implementation of Flower_Classifier to use resnet152 
    """
    def __init__(self, *args, **kwargs):
        super(Resnet152_Classifier, self).__init__(*args, **kwargs)

    def _set_model(self):
        self._model = models.resnet152(pretrained=True)
        
    def _set_input_size(self):
        self._input_size = 2048
        
    def define_classifier(self):
        self._model.fc = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(self._input_size, self._hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dp1', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(self._hidden_units, self._hidden_units)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(self._hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
    def _set_optimizer(self):
        self._optimizer = optim.Adam(self._model.fc.parameters(), lr=self._learning_rate)


class Densenet121_Classifier(Flower_Classifier):
    """
    This is a concrete implementation of Flower_Classifier to use densenet121 
    """
    def __init__(self, *args, **kwargs):
        super(Densenet121_Classifier, self).__init__(*args, **kwargs)

    def _set_model(self):
        self._model = models.densenet121(pretrained=True)
    
    def define_classifier(self):
        self._model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(self._input_size, self._hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dp1', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(self._hidden_units, self._hidden_units)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(self._hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    def _set_input_size(self):
        #1024
        self._input_size = self._model.classifier.in_features        
        
class Densenet161_Classifier(Flower_Classifier):
    """
    This is a concrete implementation of Flower_Classifier to use densenet161 
    """
    def __init__(self, *args, **kwargs):
        super(Densenet161_Classifier, self).__init__(*args, **kwargs)

    def _set_model(self):
        self._model = models.densenet161(pretrained=True)
    
    def define_classifier(self):
        self._model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(self._input_size, self._hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dp1', nn.Dropout(0.2)),
                          ('fc3', nn.Linear(self._hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    def _set_input_size(self):
        #2208
        self._input_size = self._model.classifier.in_features        
        
class Densenet169_Classifier(Flower_Classifier):
    """
    This is a concrete implementation of Flower_Classifier to use densenet169 
    """
    def __init__(self, *args, **kwargs):
        super(Densenet169_Classifier, self).__init__(*args, **kwargs)

    def _set_model(self):
        self._model = models.densenet169(pretrained=True)
    
    def define_classifier(self):
        self._model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(self._input_size, self._hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dp1', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(self._hidden_units, self._hidden_units)),
                          ('relu2', nn.ReLU()),
                          ('fc3', nn.Linear(self._hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    def _set_input_size(self):
        #1664
        self._input_size = self._model.classifier.in_features       
        
class Densenet201_Classifier(Flower_Classifier):
    """
    This is a concrete implementation of Flower_Classifier to use densenet201 
    """
    def __init__(self, *args, **kwargs):
        super(Densenet201_Classifier, self).__init__(*args, **kwargs)

    def _set_model(self):
        self._model = models.densenet201(pretrained=True)
    
    def define_classifier(self):
        self._model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(self._input_size, self._hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dp1', nn.Dropout(0.2)),
                          ('fc3', nn.Linear(self._hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    def _set_input_size(self):
        #1920
        self._input_size = self._model.classifier.in_features 

        