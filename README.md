# Image Calssifier using Deep Learning
Final project of Udacity's AI Programming with Python Nanodegree program was designed to predict flower among 102 flower categories. Learning from pre-tranined models from torchvision was transfered to achieve the goal. Models were trained using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). Dataset was loaded using torchvision and split into three train, valid, and test for training, validating, and testing the models respectfully.

# Getting started
This project was developed on Python 3.7.2. Required additional packages are NumPy, Pandas, MatplotLib, PIL, and Pytorch.

These dependencies should be install before the application can be executed. Using [conda](https://anaconda.org/anaconda/python), this packages can be intalled as below:

`conda install numpy pandas matplotlib pil`


# Training a model

## Commands to train:

Basic command: `python train.py data_directory`
Program expects data_directory to be existing with train, valid, and test subdirectories.

Optional arguments for training command:

| Argument | Exmaple |
| :------ | :----- |
| --save_dir | `python train.py data_dor --save_dir save_directory` |
|--arch|`pytnon train.py data_dir --arch vgg16`|
|--learning_rate|`python train.py data_dir --learning_rate 0.001`|
|--hidden_layer|`python train.py data_dir --learning_rate 0.001 --hidden_layer 512`|
|--epochs|`pytnon train.py data_dir --epochs 20`|
|--gpu|`python train.py data_dir --gpu`|


## Adding new model to this project

The project currently supports alexnet, vgg11, vgg13, vgg16, vgg19, resnet18, resnet34, resnet50, resnet101, resnet152, densenet121, densenet161, densenet169, and densenet201 from torchvision. To train new model, following changes need to be applied to model_factory.py.

Add a new identifier which will be used as a flag to train:
`supported_arch = ['alexnet', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'resnet18',
                  'resnet34', 'resnet50', 'resnet101', 'resnet152',
                  'densenet121', 'densenet161', 'densenet169', 'densenet201']`

Extend Flower_Classifier class to define the new model specific configuration, below is an example for Alexnet implementation:

```
class Alexnet_Classifier(Flower_Classifier):
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
```

instantiate the previously defined class in instantiate_new_model function, an example is given for alexnet implementation:
```
if in_args.arch=='alexnet':
      return set_optional_attr(in_args, Alexnet_Classifier(in_args.data_dir, Flower_Classifier.EXEC_MODES[0]))
```

# Inference a flower

## Commands to predict:

Basic command: `python predict.py /path/to/image checkpoints`

| Argument | Exmaple |
| :------ | :----- |
| --top_k | `python predict.py input checkpoint --top_k 3` |
| --category_names | `python predict.py input checkpoint --category_names cat_to_name.json` |
| --gpu | `python predict.py input checkpoint --gpu` |
