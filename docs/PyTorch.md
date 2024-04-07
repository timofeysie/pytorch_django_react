# PyTorch

## Load a pre-trained model

Here is the brief of the 

image_classification\views.py

```py
from torchvision import models
from torchvision import transforms
# load pretrained DenseNet and go straight to evaluation mode for inference
# load as global variable here, to avoid expensive reloads with each request
model = models.densenet121(pretrained=True)
model.eval()

def transform_image(image_bytes):
    """
    Transform image into required DenseNet format: 224x224 with 3 RGB channels and normalized.
    Return the corresponding tensor.
    """
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    """For given image bytes, predict the label"""
    tensor = transform_image(image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    class_name, human_label = imagenet_mapping[predicted_idx]
```

### V1 or V2

[V1 or V2? Which one should I use?](https://pytorch.org/vision/stable/transforms.html): *We recommending using the torchvision.transforms.v2 transforms instead of those in torchvision.transforms. They’re faster and they can do more things. Just change the import and you should be good to go.*

The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 0
[06/Apr/2024 15:17:18] "POST /images/ HTTP/1.1" 200 38
