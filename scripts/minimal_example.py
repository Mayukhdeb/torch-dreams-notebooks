import matplotlib.pyplot as plt
import torchvision.models as models
from torch_dreams.dreamer import dreamer

model = models.inception_v3(pretrained=True)
dreamy_boi = dreamer(model)

config = {
    "image_path": "your_image.jpg",
    "layers": [model.Mixed_6c.branch1x1],
    "octave_scale": 1.2,
    "num_octaves": 10,
    "iterations": 20,
    "lr": 0.03,
    "max_rotation": 0.5,
}

out = dreamy_boi.deep_dream(config)
plt.imshow(out)
plt.show()