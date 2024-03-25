# Pytorch Django

This project was started by following along with [Stefan Schneider's](https://stefanbschneider.github.io/blog/) article "[Using PyTorch Inside a Django App](https://stefanbschneider.github.io/blog/posts/pytorch-django/index.html)".

[PyTorch](https://pytorch.org/) is a machine learning library based on the Torch library, used for applications such as computer vision and natural language processing, originally developed by Meta AI (Facebook) and now part of the Linux Foundation umbrella. It is recognized as one of the two most popular machine learning libraries alongside [TensorFlow](https://www.tensorflow.org/).  TensorFlow, developed by Google, is also open-source and has a particular focus on training and inference of deep neural networks.
According to [Google Trends](https://trends.google.com/trends/explore?date=today%205-y&q=%2Fg%2F11gd3905v1,%2Fg%2F11bwp1s2k3&hl=en-AU), TensorFlow was more popular in 2019, but now PyTorch is more popular.  You can see why PyTorch with a focus on computer vision is a good choice to this project.

## Workflow

```py
python manage.py runserver
```

## Getting started

Create a new Django app and start the image classification module within that (I already have Django installed).

```sh
django-admin startproject pytorch_django
cd pytorch_django
python manage.py startapp image_classification
```

Install the dependencies with pip

```sh
pip install torch
pip install torchvision
```

Start the server to confirm everything is working.

```py
python manage.py runserver
Watching for file changes with StatReloader
Performing system checks...

System check identified no issues (0 silenced).

You have 18 unapplied migration(s). Your project may not work properly until you apply the migrations for app(s): admin, auth, contenttypes, sessions.
Run 'python manage.py migrate' to apply them.
March 25, 2024 - 06:26:10
Django version 3.2, using settings 'pytorch_django.settings'
Starting development server at http://127.0.0.1:8000/
```

The article uses the [DenseNet neural network](https://pytorch.org/hub/pytorch_vision_densenet/).

DenseNext is pretrained on the now famous [ImageNet dataset](https://www.image-net.org/).  This was the first fully available large-scale dataset created in 2009 that helped kickstart the current phase of deep learning in the ML age.

Now we will implement a image classification function inside the Django image_classification/views.py module.

This code is from a [PyTorch tutorial](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html) using [Flask](https://flask.palletsprojects.com/en/3.0.x/), another popular choice for Python based server apps, along with [FastApi](https://fastapi.tiangolo.com/) The code is covered under the MIT license.  I like [Django](https://www.djangoproject.com/) because it is fully featured with things like authentication and because I know all about it as I assess advanced student full stack projects using it for the [Code Institute](https://codeinstitute.net/global/full-stack-software-development-diploma/).

Stefan says: *First, I load the pretrained DenseNet, switch to evaluation/inference mode (since I do not need any further training), and load the mapping for predicted indices to human-readable labels for ImageNet. The JSON-file containing the mapping is available here and should be saved in the Django static directory under as defined in STATICFILES_DIRS (settings.py).*

Django comes with a [static files](https://docs.djangoproject.com/en/5.0/howto/static-files/) module out of the box.  The 'django.contrib.staticfiles' string should already be in the INSTALLED_APPS array in pytorch_django/settings.py:

```py
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]
```

We then store the static files in a folder called static in the app. In this case, we have to download that JSON file from the article and store it in pytorch_django/static/imagenet_class_index.json.

Now for the code.  This will go in image_classification\views.py.

Some notes on this are that we load the pretrained model once as global variable and not inside the view function, which would reload the model on each request (expensive and slow!).

And also loading the static JSON file via settings.STATIC_ROOT should work both in development and production deployment but requires running ```python manage.py collectstatic``` first.

Note if you don't run collectstatic you will see an error such as: ```TypeError: expected str, bytes or os.PathLike object, not NoneType```

```py
import io
import os
import json

from torchvision import models
from torchvision import transforms
from PIL import Image
from django.conf import settings


# load pretrained DenseNet and go straight to evaluation mode for inference
# load as global variable here, to avoid expensive reloads with each request
model = models.densenet121(pretrained=True)
model.eval()

# load mapping of ImageNet index to human-readable label (from staticfiles directory)
# run "python manage.py collectstatic" to ensure all static files are copied to the STATICFILES_DIRS
json_path = os.path.join(settings.STATIC_ROOT, "imagenet_class_index.json")
imagenet_mapping = json.load(open(json_path))
```

The ```transform_image``` function to transform an image passed in as bytes into the required format (224 x 224 RGB) for DenseNet.  It also normalizes the image, returns a tensor.

```py
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)
```

The ```get_prediction``` function is used inside the prediction function.  Here Stefan says: *the transformed tensor of the uploaded image is passed through the pretrained DenseNet model in a forward pass. The model predicts the index of the corresponding ImageNet class, which is just an integer. To display a more useful label, I retrieve the corresponding human-readable label from the imagenet_mapping dict that I created at the beginning from the downloaded JSON file*

```py
def get_prediction(image_bytes):
    """For given image bytes, predict the label using the pretrained DenseNet"""
    tensor = transform_image(image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    class_name, human_label = imagenet_mapping[predicted_idx]
    return human_label
```

The full code is in the image_classification\views.py file.

Next, create the urls image_classification/urls.py file and update the pytorch_django/urls.py.

