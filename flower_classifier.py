import os
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from abc import ABC, abstractmethod
from collections import OrderedDict
from process_image import process_image

class Flower_Classifier(ABC):
    """
    A abstract class to represent classifier flower. A concrete class must
    implement _set_input_size and _set_model methods to specific ImageNet
    model spacific implementation. Method _set_input_size should set the
    input size of the model and _set_model method should instantiate
    desired model e.g.

        self._input_size = 9216 or
        self._input_size = self._model.classifier.in_features

        self._model = models.alexnet(pretrained=True)

    Extending this class, concrete class can be used to both training and
    predicting.
    """
    #Class variable
    #Avaiable execution mode
    EXEC_MODES = ['train', 'valid', 'test']
    #Default learning rate
    DEFAULT_LEARNING_RATE = 0.003
    #Default hidden unit
    DEFAULT_HIDDEN_UNIT = 4096
    #Default number of epochs
    DEFAULT_EPOACHS = 5
    #Default checkpoint dir
    DEFAULT_CHECKPOINT_DIR = 'checkpoints'

    def __init__(self, data_dir_or_path, exec_mode):
        """
        Init method is resposible to initialize certain attributes which
        are used through out processing.

        Parameters:
            data_dir_or_path - This is path to the directory where the flowers
                            are stored
            exec_mode - This is an indicator of execution mode i.e. train or test.
                     Based on this indicator training or test specific processing
                     has been done.
        Return:
            None
        """
        super().__init__()
        self._data_dir_or_path = data_dir_or_path
        self._exec_mode = exec_mode
        self._save_dir = Flower_Classifier.DEFAULT_CHECKPOINT_DIR
        self._epochs = Flower_Classifier.DEFAULT_EPOACHS
        self._learning_rate = Flower_Classifier.DEFAULT_LEARNING_RATE
        self._hidden_units = Flower_Classifier.DEFAULT_HIDDEN_UNIT
        self._gpu=None
        self._set_model()
        self._set_input_size()
        self._set_data_transforms()
        self._set_image_datasets()
        self._set_dataloaders()

    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, save_dir_or_path):
        if os.path.isdir(save_dir_or_path):
            self._save_dir_or_path = save_dir_or_path
        else:
            raise Exception('Not a valid directory to save the checkpoint: {}'.format(save_dir))

    @property
    def input_size(self):
        return self._input_size

    @abstractmethod
    def _set_input_size(self):
        self._input_size = 25088

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        if learning_rate < 0.0:
            print('learning rate can not be negative number')
        else:
            self._learning_rate = learning_rate

    @property
    def hidden_units(self):
        return self._hidden_units

    @hidden_units.setter
    def hidden_units(self, hidden_units):
        if hidden_units < 0:
            print('hidden units can not be negative number')
        else:
            self._hidden_units = hidden_units

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, epochs):
        if epochs < 0:
            print('epochs can not be negative number')
        else:
            self._epochs = int(epochs)

    @property
    def gpu(self):
        return self._gpu

    @gpu.setter
    def gpu(self, gpu):
        self._gpu = gpu

    def _set_data_transforms(self):
        if self._exec_mode == Flower_Classifier.EXEC_MODES[0]:
            train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

            valid_transforms = transforms.Compose([transforms.Resize(255),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                                       [0.229, 0.224, 0.225])])
            test_transforms = transforms.Compose([transforms.Resize(255),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                                       [0.229, 0.224, 0.225])])
            self._data_transforms = {
                Flower_Classifier.EXEC_MODES[0]:train_transforms,
                Flower_Classifier.EXEC_MODES[1]:valid_transforms,
                Flower_Classifier.EXEC_MODES[2]:test_transforms
            }

    def _set_image_datasets(self):
        if self._exec_mode == Flower_Classifier.EXEC_MODES[0]:
            self._image_datasets = {}
            for exec_mode in Flower_Classifier.EXEC_MODES:
                self._image_datasets[exec_mode] = datasets.ImageFolder(self._data_dir_or_path+'/'+exec_mode, transform=self._data_transforms[exec_mode])

    def _set_dataloaders(self):
        if self._exec_mode == Flower_Classifier.EXEC_MODES[0]:
            self._dataloaders = {}
            for image_dataset in self._image_datasets:
                if image_dataset==Flower_Classifier.EXEC_MODES[0]:
                    self._dataloaders[image_dataset] = torch.utils.data.DataLoader(self._image_datasets[image_dataset], batch_size=64, shuffle=True)
                else:
                    self._dataloaders[image_dataset] = torch.utils.data.DataLoader(self._image_datasets[image_dataset], batch_size=64)

    @property
    def model(self):
        return self._model

    @abstractmethod
    def _set_model(self):
        self._model = models.vgg16(pretrained=True)

    def train(self):
        """
        This method is drive the training process. Unless overriden in concrete
        implementation, by default training is only available when execution mode
        is set to train.

        Method before starts training given model, freezes parameters, applies
        defined classifier, set loss criterion, and set optimizer. Then starts
        training.

        Parameters:
            self
        Return:
            None
        """
        if self._exec_mode == Flower_Classifier.EXEC_MODES[0]:
            self._freezeParameters()
            self.define_classifier()
            self._loss_criterion()
            self._set_optimizer()
            self._train_n_display_stats()
            #self.display_execution_details()
        else:
            print('Training a model is only available on train mode')

    def _freezeParameters(self):
        for param in self._model.parameters():
            param.requires_grad = False

    def define_classifier(self):
        """
        This method defines default classifier. Method must be overridden in
        concrete implementation for that specific classfication definition.

        Parameters:
            self
        Return:
            None
        """
        self._model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(self._input_size, self._hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dp1', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(self._hidden_units, self._hidden_units)),
                          ('relu2', nn.ReLU()),
                          ('dp1', nn.Dropout(0.5)),
                          ('fc3', nn.Linear(self._hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    def _loss_criterion(self):
        """
        This method sets default loss criterion. Method must be overridden in
        concrete implementation for that specific implementation.

        Parameters:
            self
        Return:
            None
        """
        self._criterion = nn.NLLLoss()

    def _set_optimizer(self):
        """
        This method sets default optimizer using predefined learning rate. Method must be overridden in
        concrete implementation for that specific implementation. learning rate can be changed in concrete
        implementation as using setter method.

        Parameters:
            self
        Return:
            None
        """
        self._optimizer = optim.Adam(self._model.classifier.parameters(), lr=self._learning_rate)

    def _train_n_display_stats(self):
        """
        This method trains a model and display training loss, validation loss, and validation accuracy.
        On completion of training, method call reposible method to save the checkpoint.

        Parameters:
            self
        Return:
            None
        """
        if self._gpu==True:
            device = "cuda"
        else:
            device = "cpu"

        self._model.to(device);

        #Training this model
        epochs = self._epochs
        print('Using device: {}'.format(device))
        for e in range(epochs):
            print("Epoch: {}/{}.. ".format(e+1, epochs))
            running_loss = 0
            for images, labels in self._dataloaders['train']:
                images, labels = images.to(device), labels.to(device)
                self._optimizer.zero_grad()
                logps = self._model.forward(images)
                loss = self._criterion(logps, labels)
                loss.backward()
                self._optimizer.step()
                running_loss += loss.item()
            else:
                validation_loss = 0
                accuracy = 0
                with torch.no_grad():
                    self._model.eval()
                    for images, labels in self._dataloaders['valid']:
                        images, labels = images.to(device), labels.to(device)

                        validation_logps = self._model(images)
                        loss = self._criterion(validation_logps, labels)
                        validation_loss += loss.item()

                        validation_ps = torch.exp(validation_logps)
                        top_p, top_class = validation_ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                self._model.train()
                print("Training Loss: {:.3f}.. ".format(running_loss/len(self._dataloaders['train'])),
                      "Validation Loss: {:.3f}.. ".format(validation_loss/len(self._dataloaders['valid'])),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(self._dataloaders['valid']) * 100))

        print("Done training")
        #Save checkpoint once the training complete
        self._checkpoint()

    def _checkpoint(self):
        """
        This method saves the checkpoint under default directory checkpoints if no other directory
        specified. Method uses .pth extension as convention to save the file. It is recommended to
        use either .pth or .pt to be able to use same class for prediction later on, unless the
        prediction logic is changed accordingly.

        Parameters:
            self
        Return:
            None
        """
        self._model.class_to_idx = self._image_datasets['train'].class_to_idx
        checkpoint = {
            'gpu_enabled':self._gpu,
            'input_size':self._input_size,
            'output_size':102,
            'hidden_input_size': self._hidden_units,
            'dropout':0.2,
            'class_to_idx': self._model.class_to_idx,
            'state_dict':self._model.state_dict()
        }
        if self._save_dir is not None or self._save_dir != '':
            torch.save(checkpoint, self._save_dir + '/' + self.__class__.__name__ + '.pth')
            print("Checkpoint has been saved at: {}".format(self._save_dir + '/' +  self.__class__.__name__ + '.pth'))

    def predict(self, image_path, topk=5):
        """
        This method predicts the class (or classes) of an image using a trained deep learning model.

        Parameters:
            self
        Return:
            None
        """
        self._model.eval()
        img = process_image(image_path)
        img.unsqueeze_(0)
        if self._gpu == True:
            img = img.to('cuda')
        logps = self._model(img)
        ps = torch.exp(logps)
        return ps.topk(topk, dim=1)

    def _display_execution_details(self):
        print('== Execution Details ==')
        #print('Model defined: ', self._model)
        #save_dir='', learning_rate=0.03, hidden_units=4096, epochs=5, gpu=None
        print('Data Directory: ', self._data_dir)
        print('Save Directory: ', self._save_dir)
        print('Learning Rate: ', self._learning_rate)
        print('Hidden Units:', self._hidden_units)
        print('Epochs: ', self._epochs)
        print('GPU: ', self._gpu)
        #print('Criterion: ', self._criterion)
        #print('Optimizer: ', self._optimizer)
