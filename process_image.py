from matplotlib.pyplot import imshow
#%matplotlib inline
from PIL import Image
import numpy as np
import torch

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    #Process a PIL image for use in a PyTorch model
    original = Image.open(image)
    original.show()
    
    #Resizing
    width, height = original.size
    aspect_ratio = float(width)/float(height)
    
    if width > height:
        pil_image = original.resize((round(256 * aspect_ratio), 256))
    elif width < height:
        pil_image = original.resize((256, round(256 / aspect_ratio)))
    else:
        pil_image = original.resize((256, 256))
    
    #Cropping
    #print(pil_image.size)
    width, height = pil_image.size
    left = width/2 - 112
    top = height/2 - 112
    right = width/2 + 112
    bottom = height/2 + 112
    pil_image = pil_image.crop((left, top, right, bottom))
    #print('left: {}, top: {}, right: {}, bottom: {}'.format(left, top, right, bottom))
    
    #print(len(np.array(pil_image)))
    np_image = np.array(pil_image) / 255
    
    #Normalizing image
    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    #Transposing leaving second dimension
    np_image = np_image.transpose((2, 0, 1))
    return torch.from_numpy(np_image).type(torch.FloatTensor)