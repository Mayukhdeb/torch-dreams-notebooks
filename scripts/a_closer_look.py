import matplotlib.pyplot as plt
import torchvision.models as models
from torch_dreams.dreamer import dreamer

model = models.inception_v3(pretrained=True)

"""
torch_dreams.dreamer is basically a wrapper over any PyTorch model,
it would enable us to optimize the input image to activate various feature(s) within the 
neural network
"""

dreamy_boi = dreamer(model)

"""
The config is where we get to customize how exactly we want the optimization to happen. 

* image_path specifies the relative path to the input image 
* layers: This is a list where you pass the layers whose outputs are to be "stored" for optimization layer on. 
* octave_scale: The algorithm in torch_dreams resizes the input image iteratively from size (original size)/(octave_scale**n) to the original size. 
  This is reminiscent of the "octave scale" used by Alexander Mordvintsev in his DeepDream Tensorflow tutorial. 
* num_octaves: specifies the number of times the image is scaled up in order to reach back to the original size while running the algorithm. 
* iterations: Number of gradient ascent steps taken per octave. 
* lr: Learning rate used in each step of the gradient ascent. 
* custom_func: Use this to build your own custom optimization functions to optimize on individual channels/units/etc. More on this later. By default, it will optimize the input image on all of the layers mentioned in layers. 
* max_rotation: Caps the maximum rotation to apply on the image before each gradient ascent step. Rotation transforms helped in reducing high frequency noise.
* gradient_smoothing_coeff: Use this to apply Gaussian blur to the gradients before the gradient ascent step. 
  Ideal values range around 0.5 if used (higher value-> stronger blur). Useful to remove high frquency patterns sometimes. 
* gradient_smoothing_kernel_size: Kernel size to be used when applying gaussian blur. 

"""
config = {
    "image_path": "your_image.jpg",
    "layers": [model.Mixed_6c.branch1x1],
    "octave_scale": 1.2,
    "num_octaves": 10,
    "iterations": 20,
    "lr": 0.03,
    "custom_func": None,
    "max_rotation": 0.5,
    "gradient_smoothing_coeff": 0.1,
    "gradient_smoothing_kernel_size": 3
}

config["layers"] = [
  model.Mixed_6d.branch1x1,
  model.Mixed_5c
]

out = dreamy_boi.deep_dream(config)
plt.imshow(out)
plt.show()