"""
Created on Thu Oct 26 11:19:58 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
from torch.autograd import Variable
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt

#from scipy.misc import imread, imsave
from PIL import Image

from misc_functions import get_params, convert_to_grayscale, save_gradient_images
from pytorch_utils import get_model, set_gpu, get_pytorch_module, get_input_size, \
        get_transform_detransform, get_short_imagenet_name, get_first_module_name


class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model, layer=None, size=224):
        self.model = model
        # If no layer is specified, use the first layer (i.e., input layer)
        if layer is None:
            layer = get_first_module_name(model)
        self.layer = layer
        self.module = get_pytorch_module(model, self.layer)
        self.up = nn.Upsample(size=size, mode='bilinear')
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to save gradient
        self.module.register_backward_hook(hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        if input_image.is_cuda:
            one_hot_output = torch.cuda.FloatTensor(1, model_output.size()[-1]).zero_()
        else:
            one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        upsampled_grads = self.up(self.gradients)
        return np.max(np.abs(upsampled_grads.data.cpu().numpy()[0]), axis=0)

def run_vanilla_backprop(image='../input_images/snake.jpg', 
        arch='alexnet', dataset='imagenet', layer=None, target_class=None, 
        visualize=True, cuda=False):
    # Get pretrained model
    model = get_model(arch, pretrained=True, cuda=cuda)
    # Get input size
    size = get_input_size(dataset=dataset, arch=arch)[-1]
    # Vanilla backprop
    VBP = VanillaBackprop(model, layer=layer, size=size)
    # Get pytorch transform (for data normalization/denormalization)
    (transform, detransform) = get_transform_detransform(dataset=dataset, size=size, train=False)
    # Load image
    if isinstance(image, str):
        img_ = transform(Image.open(image).convert('RGB')).unsqueeze(0)
    else:
        img_ = image
    img = Variable(img_.cuda() if cuda else img_, requires_grad=True)
   # If no target class is given, use the top predicted class
    if target_class is None:
        target_class = np.argmax(model(img).data.cpu().numpy()[0])
    if dataset == 'imagenet':
        print('Visualizing imagenet class %s (%d)' % 
                (get_short_imagenet_name(target_class), target_class))
    else:
        print('Visualizing %s class index %d' % (dataset, target_class)) 

    # Generate gradients
    vanilla_grads = VBP.generate_gradients(img, target_class)

    if visualize:
        f, ax = plt.subplots(1,1)
        ax.imshow(detransform(img_[0]))
        ax.imshow(vanilla_grads, alpha=0.5, cmap='jet')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
    # Save colored gradients
    #save_gradient_images(vanilla_grads, file_name_to_export + '_Vanilla_BP_color')
    # Convert to grayscale
    #grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
    # Save grayscale gradients
    #save_gradient_images(grayscale_vanilla_grads, file_name_to_export + '_Vanilla_BP_gray')
    #print('Vanilla backprop completed')
    return vanilla_grads

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TODO')

    parser.add_argument('--image', type=str, default='TODO')
    parser.add_argument('--arch', type=str, default='alexnet')
    parser.add_argument('--dataset', type=str, default='imagenet')
    parser.add_argument('--layer', type=str, default=None)
    parser.add_argument('--target_class', type=int, default=None)
    parser.add_argument('--gpu', type=int, nargs='*', default=None)

    args = parser.parse_args()

    cuda = set_gpu(args.gpu)

    run_vanilla_backprop(image_path=args.image, arch=args.arch, layer=args.layer,
            target_class=args.target_class, cuda=cuda)
