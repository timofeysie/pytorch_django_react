# PyTorch, Django & React

This is a fullstack machine learning app to demonstrate the basics roles of a ML engineer.

The current PyTorch functions are based on [Stefan Schneider's](https://stefanbschneider.github.io/blog/) article "[Using PyTorch Inside a Django App](https://stefanbschneider.github.io/blog/posts/pytorch-django/index.html)".  This covers using a pre-trained model for image classification and a simple html page to upload an image and put it to work.

I have implemented a [Django REST Framework API](https://www.django-rest-framework.org/) to expose the use of the model to create a more real-world example of using a [React](https://react.dev/) frontend deploy ML projects.  This project is in a separate repo called [PyTorch Frontend](https://github.com/timofeysie/pytorch_frontend).

This is just a basic demonstration deploying a pre-trained model for use on the web.  For me, a full stack ML engineer is a role that needs to have an awareness of Data Science and the AI landscape, but doesn't have to be an expert in all the fields.

This project covers just some of the roles that I consider a fullstack ML engineer may be responsible for.  Things that could also be considered part of this role that I am not covering at the moment include:

- Designing AI models
- Data management and pipelines
- Infrastructure and architecture
- Testing and validation

In the future I hope to also include demonstrations of some of these skills so stay tuned!

## Library details

[PyTorch](https://pytorch.org/) is a machine learning library based on the Torch library, used for applications such as computer vision and natural language processing, originally developed by Meta AI (Facebook) and now part of the Linux Foundation umbrella. It is recognized as one of the two most popular machine learning libraries alongside [TensorFlow](https://www.tensorflow.org/).  TensorFlow, developed by Google, is also open-source and has a particular focus on training and inference of deep neural networks.

According to [Google Trends](https://trends.google.com/trends/explore?date=today%205-y&q=%2Fg%2F11gd3905v1,%2Fg%2F11bwp1s2k3&hl=en-AU), TensorFlow was more popular in 2019, but now PyTorch is more popular.  You can see why PyTorch with a focus on computer vision is a good choice to this project.

The React app was started with [Vite](https://vitejs.dev/guide/) and uses the [Material UI](https://mui.com/) component library.

## Project Workflow Reference

```sh
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
pip freeze > requirements.txt
python manage.py runserver
```

## Getting started

Create a new Django app and start the image classification module within that.

```sh
django-admin startproject pytorch_django
cd pytorch_django
python manage.py startapp image_classification
```

Note I already have Django installed.  This project was created with my local version which is Django 3.2.

The [repo from the article](https://github.com/stefanbschneider/pytorch-django/blob/main/pytorch_django/settings.py) is using Django 3.1.4.  You can install this version using ```pip install 'django<4'```.  I'm not sure how these code examples with work with the latest version of Django.

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

## Creating a DRF API

I have used a few sources to learn Django REST Framework, but [the official tutorial](https://www.django-rest-framework.org/tutorial/1-serialization/#tutorial-1-serialization) is a good a place as any.  I will cover just the details required for our purpose here.

### Install the the Django REST Framework

```sh
pip install djangorestframework
```

Lets also create an another app for the REST endpoints:

```sh
python manage.py startapp images
```

Add these two new app after the static files app at the bottom of the installed apps array in settings.py:

```py
INSTALLED_APPS = [
    ...
    'django.contrib.staticfiles',
    'rest_framework',
    'images',
]
```

### The Image Model

The Image model looks like this:

```py
from django.db import models

class Image(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=100, blank=True, default='')

    class Meta:
        ordering = ['created']
```

A Meta class is created that will return a Image instances where the most recently created is first.  

Register the Image model in admin.py

```py
from django.contrib import admin
from .models import Image

admin.site.register(Image)
```

Then run make migrations which you have to do after updating a model:

```shell
python manage.py makemigrations
python manage.py migrate
```

Create a admin user and provide a password:

```shell
python manage.py createsuperuser
```

Run the server:

```shell
python manage.py runserver
```

Goto the admin url: http://127.0.0.1:8000/admin

Create a file with the dependencies:

```shell
pip freeze > requirements.txt
```

### Import the APIView and Response classes in views.py

Images will extend APIView similar to Django's View class.

It also provides a few bits of extra functionality such as making sure to receive a Request instances in the view, handling parsing errors, and adding context to Response objects.

Create the Images view and define the get method.

profiles\views.py

```py
from rest_framework.views import APIView
from rest_framework.response import Response
from .models import Image

class Images(APIView):
    def get(self, request):
        images = Image.objects.all()
        return Response(images)
```

The GET at this point wont really be used because we will not be adding images to any db table yet.  A more usable app would have a list of images and their guesses by user, but this would require authentication, cloud storage and more.  To keep things simple for now, we will just focus on implementing the POST API and save the auth and db/image storage work for a later article.

### Create images urls

Create a urls.py file with the images path.

```py
from django.urls import path
from images import views

urlpatterns = [
    path('images/', views.images.as_view()),
]
```

Include profile urls in the main apps urls.py

drf_two\urls.py

```py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('images.urls')),
]
```

Currently if you start the server and go to 'http://127.0.0.1:8000/images/' there is an error:

```err
TypeError at /images/
Object of type Image is not JSON serializable
```

This can be fixed by creating a serializer.

### Create the serializer

Creating serializers.py, import serializers from rest framework and our Profile model.

Specify 'owner' as a ReadOnlyField and populate it with the owner's username.

In the Meta class point to the Profile model and specify the fields we want in the response.

images\serializers.py

```py
from rest_framework import serializers
from .models import Image

class ImageSerializer(serializers.ModelSerializer):
    owner = serializers.ReadOnlyField(source='owner.username')

    class Meta:
        model = Image
        # to list all fields all in an array or set to '__all__'  
        fields = [
            'id', 'owner', 'created_at', 'updated_at', 'title',
        ]
```

When extending Django's model class using models.models, the id field is created automatically

### Add the serializer to the views.py file

Import the ImageSerializer, create a ImageSerializer instance and pass in images and many equals True to specify serializing multiple Image instances.
In the Response send data returned from the serializer.

images\views.py

```py
...
from .serializers import ImageSerializer

class Images(APIView):
    def get(self, request):
        images = Image.objects.all()
        serializer = ImageSerializer(images, many=True)
        return Response(serializer.data)
```

Now the JSON user list is returned.

Next, we need to create a POST endpoint to receive the image form and return the result.

Again we do that in images/views.py

```py
...
from rest_framework import status
from image_classification.forms import ImageUploadForm
from image_classification.views import get_prediction

class Images(APIView):
    ...

    def post(self, request):
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_bytes = image.file.read()

            # Process the image and get the predicted label
            try:
                predicted_label = get_prediction(image_bytes)
            except RuntimeError as re:
                print(re)
                predicted_label = "Prediction Error"

            # Return the predicted label in the response
            return Response({'predicted_label': predicted_label})
        else:
            return Response(form.errors, status=status.HTTP_400_BAD_REQUEST)
```

To test this POST endpoint there are a few options.  One would be to jump in with a React frontend app.  True we already have an index.html form served in the root of the app, but we want to test our new POST DRF API endpoint.  We know the get_prediction function works.

Another choice would be to use Postman or automated e2e testing with Cypress.  A unit test is another good option.  But the simplest way would be a cURL statement from the command line:

```sh
curl -X POST -F "image=@static/test-image.png" http://localhost:8000/images
```

I like to capture all the errors I run into during development in my notes in case it helps later.  The above causes this error:

```err
RuntimeError: You called this URL via POST, but the URL doesn't end in a slash and you have APPEND_SLASH set. Django can't redirect to the slash URL while maintaining POST data. Change your form to point to localhost:8000/images/ (note the trailing slash), or set APPEND_SLASH=False in your Django settings.
```

So live an learn.  Add a slash to the end of the url:

```sh
$ curl -X POST -F "image=@static/test-image.png" http://localhost:8000/images/
{"predicted_label":"Prediction Error"}
```

That's funny, I thought I tested this image and it worked in the demo webpage.  This first test image is actually a screenshot of an image of two African elephants I downloaded from Wikimedia.  The original download says its filetype is html which is why I took a screenshot to have a smaller png version of it,

The original download title: Getty_equivocation-538631923-5755ce355f9b5892e8d88173.webp

I will rename this as test-image-2.webp and give it a shot.

```sh
$ curl -X POST -F "image=@static/test-image-2.webp" http://localhost:8000/images/
{"predicted_label":"African_elephant"}
```

Time to test some more images.  I am going to use the [Wikimedia Commons site](https://commons.wikimedia.org/wiki/Category:Images) which are free to use for whatever being covered under the Creative Commons license.

[This image of a teapot](https://commons.wikimedia.org/wiki/File:Teapot_Featuring_Pink_and_Diamond_Patterns_around_the_Base_-_DPLA_-_c949a251fd4c60e55b375a64fd9debcf.jpg) looks good.  I have a teapot at home I can also try out with my own picture to try something original because there is a good chance Getty images were used to train our model.

Rename the image and try again:

```sh
$ curl -X POST -F "image=@static/test-image-3.jpg" http://localhost:8000/images/
{"predicted_label":"teapot"}
```

As a side note on the error above, apparently Django has a flag to disable the APPEND_SLASH setting in your Django settings. To do this, open your Django project's settings.py file and set APPEND_SLASH to False:

```py
APPEND_SLASH = False
```

After making this change, you can use the original cURL command without the trailing slash:

Anyhow, we know our endpoint works now.

The next step is to actually create a React app that can be deployed and work in production.  Then we can start adding ML functionality to it to start to create something useful.

## The React frontend

It is possible to create a react app in a frontend directory in this project, however I am not a fan of combining the two as there is a chance that one can effect the other.  I like separately deployable frontend and backend projects.  Yes this article is about fullstack ML engineering, but even such a role needs to work within a team of specialists.  I want a designer skilled as CSS to jump into the frontend code and do their stuff without having to worry about the pytorch_django directory.  I also want a data scientist to be able to get started with training a model in the image_classification directory without worrying about the React code.  This is just my opinion, but its based on real work lessons from team projects.

Starting a React app used to be a simple thing.  Everyone used CRA (create-react-app) and moved on.  However, with the official documentation released last year, CRA is no longer recommended.  The new [Start a New React Project](https://react.dev/learn/start-a-new-react-project) documentation instead offers a rage of suggestions.  I wont go into the politics of this list

- Next.js
- Remix
- Gatsby
- Expo (for native apps)
- Bleeding-edge React framework features like React Server Components.
- Next.js (App Router)

There are also two mentioned as custom build process with a bundler (which is what CRA was I suppose):

- Vite
- Parcel

I have a lot of experience with Next.js and Vite.  I would like to try Expo for this project, but that would add a whole other layer of layer of complexity involved with running and deploying hybrid mobile apps.  I've done a lot of work in the past with Ionic, an Angular based hybrid framework, and I don't miss dealing with the app stores at all.

I think to keep things simple for this tutorial I will choose [Vite](https://vitejs.dev/guide/).

Next, to TypeScript or not?  If this was going to be a big project with serious goals I would for sure use TypeScript.  However, since this is an educational project, I think the less complexity there is on the table, the better, so we will go with vanilla JavaScript for now.  Also, you shouldn't automatically reach for extra tooling that does not provide an added purpose.  This article can always be updated in the future if there is good reason to use TypeScript or any other libraries that will improve it.

Finally, the UI library.  To save time styling things to perfection, I want to use a UI framework.  The [top choices at the moment](https://trends.google.com/trends/explore?date=today%205-y&q=chakra%20ui,next%20ui,mui,bootstrap,%2Fg%2F11mw737tbt&hl=en-GB) are:

- [TailwindCss](https://tailwindcss.com/)
- [React-Bootstrap](https://react-bootstrap.netlify.app/)
- [Material UI](https://mui.com/)
- [Chakra](https://chakra-ui.com/)
- [NextUI](https://nextui.org/)

There are a lot of opinions regarding these which I wont go into here, but I would like to justify my choice a tiny bit.  I use MUI for work, so for this reason it will save time for me here, as well as feed into making my work life easier.  If I wanted to add another framework to my resume, I might choose Tailwind, as I have only a small amount of experience with it.  I have used Bootstrap a ton, so been-there, done-that, not going back unless forced.

```sh
npm create vite@latest pytorch-frontend -- --template react
Need to install the following packages:
  create-vite@5.2.3
Ok to proceed? (y) y
npm WARN EBADENGINE Unsupported engine {
npm WARN EBADENGINE   package: 'create-vite@5.2.3',
npm WARN EBADENGINE   required: { node: '^18.0.0 || >=20.0.0' },
npm WARN EBADENGINE   current: { node: 'v16.20.0', npm: '8.19.4' }
npm WARN EBADENGINE }

Scaffolding project in C:\Users\timof\repos\react\pytorch-frontend...

Done. Now run:

  cd pytorch-frontend
  npm install
  npm run dev
```

Next, lets [install MUI](https://mui.com/material-ui/getting-started/installation/) and some icons.

```sh
npm install @mui/material @emotion/react @emotion/styled
npm install @mui/icons-material
```

### The design

First we should think about how the component should look.

We want two buttons, one to choose an image from the file system, and one to POST the image to the backend.

Then we display the predicted_label as a caption when it returns.

Its a good idea to sketch a wire frame drawing to see if any issues come up before you commit to coding.

To save time on this, it helps to choose stock features unless a dedicated designer is available.  Projects can waste a lot of time creating custom components that don't add value to the user.  I would recommend looking at the big [Material UI componentsMaterial UI components](https://mui.com/material-ui/all-components/) page and using something there before branching out to create something new, unless it's something new that will distinguish a project.  Here the focus is on fullstack ML engineering tasks, so there is not much design-wise that would add to that except where to point out what skills are needed for this role.

Here I will use a bit of a mix of MUI and custom work.  I think its a good idea for fullstack devs to keep their CSS skills sharp and don't let them slip away.

### The layout

We can start with just a basic component and build upon that after we have something on the screen.

We don't want to put any specific work into the App.jsx, so lets create a components directory with a component for the image classification upload feature.

src\components\ImageClassification.jsx

```js
import { Container, IconButton } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";

const ImageClassification = () => {
  return (
    <>
      <div>ImageClassification</div>
      <Container maxWidth="sm">
        <IconButton>
          <CloudUploadIcon sx={{ color: "black" }} />
        </IconButton>
      </Container>
    </>
  );
};

export default ImageClassification;
```

I used a React snippet "rafce" to generate that.

Then I start typing "<Conta...>" a menu shows up and I can select Container from it and the import will appear for me.

I do the same with the CloudUploadIcon and set the color to black.

Then I open the VSCode command palette and format the file with Prettier.

We can then use this in our App.jsx file and replace the default content with our new component.

```js
import ImageClassification from "./components/ImageClassification";

function App() {
  return (
    <>
      <ImageClassification />
    </>
  );
}
```

If you didn't run ```npm i```, then do that now, and then start the demo server with ```npm run dev```.

Now we have a button, we can make it call our call our API.

### Making API calls

To make API calls in React, we often use [Axios](https://axios-http.com/): *Promise based HTTP client for the browser*

Import axios into the frontend:

```sh
npm i axios
```

Create an src/api/axiosDefaults.js

```js
import axios from "axios";

axios.defaults.baseURL = "http://127.0.0.1:8000/";
axios.defaults.headers.post["Content-Type"] = "multipart/form-data";
axios.defaults.withCredentials = true;

export const axiosReq = axios.create();
export const axiosRes = axios.create();
```

Instead of using axios directly like this ```axios.post("images/", payload);```, we use these default objects like this:

```js
import { axiosReq, axiosRes } from "../api/axiosDefaults";
```

```js
    try {
      const { data } = await axiosReq.post("/images/", payload);
      // do something
    } catch (err) {
      console.log(err);
    }
```

### The image GET and CORS

To start things off, we will just test out calling the API from our component.

This really does not belong in the ImageClassification component, but before we get to the POST, lets make sure things are working.  We can always extract the functionality needed for a list of previous images into a separate component later.

Often a component needs to get its data when it's first loaded.  This is how we would use axios to do that.

```js
const [imageList, setImageList] = useState();

 const handleMount = async () => {
    try {
      const { data } = await axios.get("images/");
      setImageList(data);
    } catch (err) {
      console.log(err);
    }
  };

  useEffect(() => {
    handleMount();
  }, []);
```

If you run this now you would notice the call failing with a message like this in the console:

```err
Access to XMLHttpRequest at 'http://localhost:8000/images/' from origin 'http://127.0.0.1:5173' has been blocked by CORS policy: The value of the 'Access-Control-Allow-Credentials' header in the response is '' which must be 'true' when the request's credentials mode is 'include'. The credentials mode of requests initiated by the XMLHttpRequest is controlled by the withCredentials attribute.
```

CORS stands for Cross-origin resource sharing, and is the bane of FEDs (Front-end developers in case you didn't know).

To fix this, we need to add whatever URL we will be calling the backend from in the array of ALLOWED_HOSTS in pytorch_django\settings.py.

In this case running locally we add localhost as well as set CORS_ALLOW_CREDENTIALS.

We also want to handle this properly, as these things will change once deployed.  So we need to create a env.py in the root of the directory and add CLIENT_ORIGIN key:

```py
import os
os.environ['CLIENT_ORIGIN'] = 'http://127.0.0.1:5173'
```

Then in pytorch_django\settings.py we first import these settings:

```py
if os.path.exists('env.py'):
    import env
```

And after the DEBUG add the ALLOWED_HOSTS array add the localhost to ALLOWED_HOSTS:

```py
# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ['localhost']

# Application definition

INSTALLED_APPS = [ ... ]
```

Notice that security warning.  That's very important.  We will deal with that later when we prepare for the deployment.

We will also need the django-cors-headers package so install that if you haven't already done so:

```sh
pip install django-cors-headers
```

Then we need to add it to the MIDDLEWARE array and add some logic for CORS_ALLOWED_ORIGINS after that:

```py
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    ...
]

if 'CLIENT_ORIGIN' in os.environ:
    CORS_ALLOWED_ORIGINS = [
        os.environ.get('CLIENT_ORIGIN')
    ]
else:
    CORS_ALLOWED_ORIGINS = [
        'http://localhost:3000',
    ]
```

When preparing for deployment we can add values here for the deployed URLS.

If you run the app now, you should see the /images GET being called.

Next we need to do some more work to make the POST happen.

### The image input functionality

It's tempting to reach for a pre-made component such as this [MUI file input](https://viclafouch.github.io/mui-file-input/).

But I'd like to walk through creating a custom solution so we can talk about extracting it as a re-usable component.  I have plans for this app which will require choosing files in other places.  We will pretend like there are no good options and proceed with making out own.

The regular unstyled HTML ```<input type="file">``` looks like a raw HTML button and is not going to impress anyone, so I have added custom styles to it.

To accomplish this I created a CSS module in the same directory.  Many people create a separate files directory with only CSS but I like to co-locate them to where they are used so they are just a click away.

```txt
src\components\ImageClassification.jsx
src\components\ImageClassification.module.css
```

The styles then look like this:

```css
.input {
    display: none;
}

.custom_file_upload {
  height: 30px;
  width: 30px;
  display: inline-block;
  cursor: pointer;
  margin: 6px;
}
```

Now, to use them, you can import them into the component and use the class needed using dot notation.

```js
import styles from "./ImageClassification.module.css";

...
<label
    htmlFor="contained-button-file".1
    className={styles.custom_file_upload}
    >
    <FolderIcon sx={{ color: "black" }} />
    <input
        accept="image/*"
        className={styles.input}
        id="contained-button-file"
        multiple
        type="file"
        onChange={onSelectFile}
        ref={fileInputRef}
    />
</label>
```

My idea initially was to have an image container the same size as the image will be transformed into on the backend (224 x 224), then if there is no image, show a button to choose the file.  However, since this is a native element, I couldn't get the same event handler to work for the image as the input, so I have gone with separate controls discussed in the next section.

The JavScript for the image file chooser input looks like this:

```js
  const [selectedFile, setSelectedFile] = useState();
  const [preview, setPreview] = useState([]);
  const fileInputRef = useRef(null);

  /**
   * Clear the predicted label and set the selected file object.
   */
  const onSelectFile = (e) => {
    if (!e.target.files || e.target.files.length === 0) {
      setSelectedFile(undefined);
      return;
    }
    setPredictedLabel("");
    setSelectedFile(e.target.files[0]);
  };

  /**
   * Create a preview as a side effect whenever selected file is changed.
   */
  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
    // free memory when ever this component is unmounted
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

```

### The image POST

There are two parts to this.  First, there is an event handler for when the user selects an image using the input file chooser shown above.

```js
  const [predictedLabel, setPredictedLabel] = useState("(select & upload to classify)");

  /**
   * Clear the predicted label and set the selected file object.
   */
  const onSelectFile = (e) => {
    if (!e.target.files || e.target.files.length === 0) {
      setSelectedFile(undefined);
      return;
    }
    setPredictedLabel("");
    setSelectedFile(e.target.files[0]);
  };
```

The second is an event handler for when the user chooses the upload & classify button.  Async actions like this should be in a try/catch block which for now will just print out any error to the console.

```js
  /**
   * Post the selected image and set the predicted label with the result.
   * @param {*} event
   * @returns
   */
  const handleUploadImage = async (event) => {
    event.preventDefault();
    setPredictedLabel("");
    if (!selectedFile) {
      return;
    }
    try {
      const formData = new FormData();
      formData.append("image", selectedFile);
      const response = await axiosReq.post("/images/", formData);
      // Set the predicted label based on the response
      setPredictedLabel(response.data.predicted_label);
    } catch (error) {
      console.log(error);
    }
  };
```

The [formData object instance](https://developer.mozilla.org/en-US/docs/Web/API/FormData) provides a way to construct a set of key/value pairs representing form fields and their values, which designed to be sent using the fetch(), XMLHttpRequest.send() or navigator.sendBeacon() methods. It uses the same format a form would use if the encoding type were set to "multipart/form-data".

We use async/await to pause and set the predicted label when it arrives.

### The full layout

Putting it all together, I used MUI layout components like Stack, Container and Typography.  A lot more can be done to perfect this, but I'm not sure if this is the final design.  As they say, don't optimize something that shouldn't exist.

I used the styled function to create the Item component.  This was straight out of the docs.  If this project keeps going, it's worth creating a theme to make things more consistent and the code cleaner.

```js
import { styled } from "@mui/material/styles";

...

  const Item = styled(Paper)(({ theme }) => ({
    backgroundColor: theme.palette.mode === "dark" ? "#1A2027" : "#fff",
    ...theme.typography.body2,
    padding: theme.spacing(1),
    textAlign: "start",
    color: theme.palette.text.secondary,
    minWidth: 300,
    width: 300,
    justifyContent: "start",
  }));

  return (
    <>
      <Stack>
        <Typography noWrap>Image Classification</Typography>
        <Container>
          <div className={styles.image_container}>
            {selectedFile ? (
              <label htmlFor="contained-button-file">
                <img
                  src={preview}
                  className={styles.preview}
                  onClick={handleClickImage}
                />
              </label>
            ) : (
              <p>Choose image</p>
            )}
          </div>
        </Container>
        <Item
          sx={{
            my: 1,
            mx: "auto",
            minWidth: "224px",
          }}
        >
          <div>Label: {predictedLabel}</div>
        </Item>
        <Item
          sx={{
            my: 1,
            mx: "auto",
            display: "flex",
            justifyContent: "start",
            alignItems: "center",
            textAlign: "start",
            maxWidth: "300px",
          }}
        >
          <label
            htmlFor="contained-button-file"
            className={styles.custom_file_upload}
          >
            <FolderIcon sx={{ color: "black" }} />
            <input
              accept="image/*"
              className={styles.input}
              id="contained-button-file"
              multiple
              type="file"
              onChange={onSelectFile}
              ref={fileInputRef}
            />
          </label>
          <Typography ml={1} sx={{ textOverflow: "ellipsis" }}>
            {selectedFile?.name}
          </Typography>
        </Item>
        <Item
          sx={{
            my: 1,
            mx: "auto",
            p: 2,
            minWidth: "224px",
            display: "flex",
            flexDirection: "row",
            alignItems: "center",
          }}
        >
          <IconButton sx={{ marginTop: "5px" }} onClick={handleUploadImage}>
            <CloudUploadIcon sx={{ color: "black" }} />
          </IconButton>
          <Typography>Upload & classify</Typography>
        </Item>
      </Stack>
    </>
  );
```

You can get an idea here of how MUI elements are styled.

One-off customization for specific MUI components can be styled with CSS using the ```sx={{ ... }}``` attribute.  Notice CSS properties are now camelCase and not snake-case.

The docs for the [sx prop are here](https://mui.com/material-ui/react-box/).

MUI also has Layout utilities

```jsx
<Box mt={1} justifyContent="flex-end">
```

```mt={2}``` is a shorthand form of 'margin-top'.  Check out the list of [shorthand utility classes here](https://mui.com/system/spacing/).

### Full code

This is still a work in progress, but in its current state, the component looks like this:

```js
import { Container, IconButton, Stack } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import FolderIcon from "@mui/icons-material/Folder";
import styles from "./ImageClassification.module.css";
import { useEffect, useState, useRef } from "react";
import { axiosReq } from "../api/axiosDefaults";
import { styled } from "@mui/material/styles";
import Paper from "@mui/material/Paper";
import Typography from "@mui/material/Typography";

const ImageClassification = () => {
  const [selectedFile, setSelectedFile] = useState();
  const [preview, setPreview] = useState([]);
  const [predictedLabel, setPredictedLabel] = useState("<select & classify>");
  const fileInputRef = useRef(null);

  /**
   * Post the selected image and set the predicted label with the result.
   * @param {*} event
   * @returns
   */
  const handleUploadImage = async (event) => {
    event.preventDefault();
    setPredictedLabel("");
    if (!selectedFile) {
      return;
    }
    try {
      const formData = new FormData();
      formData.append("image", selectedFile);
      const response = await axiosReq.post("/images/", formData);
      // Set the predicted label based on the response
      setPredictedLabel(response.data.predicted_label);
    } catch (error) {
      console.log(error);
    }
  };

  const onSelectFile = (e) => {
    if (!e.target.files || e.target.files.length === 0) {
      setSelectedFile(undefined);
      return;
    }
    setPredictedLabel("");
    setSelectedFile(e.target.files[0]);
  };

  /**
   * Create a preview as a side effect whenever selected file is changed.
   */
  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
    // free memory when ever this component is unmounted
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const handleClickImage = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const Item = styled(Paper)(({ theme }) => ({
    backgroundColor: theme.palette.mode === "dark" ? "#1A2027" : "#fff",
    ...theme.typography.body2,
    padding: theme.spacing(1),
    textAlign: "start",
    color: theme.palette.text.secondary,
    minWidth: 300,
    width: 300,
    justifyContent: "start",
  }));

  return (
    <>
      <Stack>
        <Typography noWrap>Image Classification</Typography>
        <Container>
          <div className={styles.image_container}>
            {selectedFile ? (
              <label htmlFor="contained-button-file">
                <img
                  src={preview}
                  className={styles.preview}
                  onClick={handleClickImage}
                />
              </label>
            ) : (
              <p>Choose image</p>
            )}
          </div>
        </Container>
        <Item
          sx={{
            my: 1,
            mx: "auto",
            minWidth: "224px",
          }}
        >
          <div>Label: {predictedLabel}</div>
        </Item>
        <Item
          sx={{
            my: 1,
            mx: "auto",
            display: "flex",
            justifyContent: "start",
            alignItems: "center",
            textAlign: "start",
            maxWidth: "300px",
          }}
        >
          <label
            htmlFor="contained-button-file"
            className={styles.custom_file_upload}
          >
            <FolderIcon sx={{ color: "black" }} />
            <input
              accept="image/*"
              className={styles.input}
              id="contained-button-file"
              multiple
              type="file"
              onChange={onSelectFile}
              ref={fileInputRef}
            />
          </label>
          <Typography ml={1} sx={{ textOverflow: "ellipsis" }}>
            {selectedFile?.name}
          </Typography>
        </Item>
        <Item
          sx={{
            my: 1,
            mx: "auto",
            p: 2,
            minWidth: "224px",
            display: "flex",
            flexDirection: "row",
            alignItems: "center",
          }}
        >
          <IconButton sx={{ marginTop: "5px" }} onClick={handleUploadImage}>
            <CloudUploadIcon sx={{ color: "black" }} />
          </IconButton>
          <Typography>Upload & classify</Typography>
        </Item>
      </Stack>
    </>
  );
};

export default ImageClassification;
```

Here is the associated css module: src\components\ImageClassification.module.css

```css
.input {
    display: none;
}

.custom_file_upload {
  height: 30px;
  width: 30px;
  display: inline-block;
  cursor: pointer;
  margin: 6px;
}

.image_container {
  min-height: 224px;
  max-width: 300px;
  width: auto;
  border: dashed;
  display: flex;
  justify-content: center;
  align-items: center;
}

.preview {
  height: 224px;
}
```

### Deploying the modal

To start with, we will simply be using Heroku.  This would probably also work just as well with Vercel or other free/paid web hosts.  All we need at this point is web hosting.

### ML cloud providers landscape

I will just discuss a brief overview of other deployment options.  For starters there are the big three:

- [Azure](https://portal.azure.com/)
- [AWS](https://aws.amazon.com/)
- [Google Cloud](https://cloud.google.com/)

There is a lot more to ML engineering besides just a web host.  When doing ML fullstack engineering professionally it would be good to have experience with a cloud provider to train models, possibly in a distributed fashion, while recording training and evaluation metrics, etc.

Azure has the [Machine Learning Studio](https://azure.microsoft.com/en-us/products/machine-learning) which Have enjoyed using in the past but it got quite expensive when you start paying for machine time to train models.

With AWS, I have used Cognito and of course S3 buckets quite a lot.  I have never used their ML tools, and it would be interesting to see what their options are like these days.  I keep hearing about [SageMaker](https://aws.amazon.com/pm/sagemaker) to it would be good to have some experience with that also.

Google also has a lot of [AI and machine learning products](https://cloud.google.com/products/ai) these days.  I have used [Google Colab](https://colab.research.google.com/) for data science though I usually prefer to run [Jupyter notebooks](https://jupyter.org/) locally on my laptop.

I think it all depends on the company you might work for and what your goals are.  You can't do everything, so it does pay to specialize.

It always helps to have knowledge of popular infrastructure choices like the following (advertising blurbs included):

- [IaC](https://mlopsnow.com/blog/4-infrastructure-as-code-services-that-can-supercharge-your-ml-infrastructure/) (Infrastructure-as-Code): *For those serious about MLOps*
- [Vertex AI](https://cloud.google.com/vertex-ai) Google Cloud: *everything you need to build and use generative AI*
- [Algorithmia](https://www.datarobot.com/platform/mlops/?redirect_source=algorithmia.com): *Your Center of Excellence for Machine Learning Operations and Production AI*
- [AWS EC2](https://aws.amazon.com/pm/ec2/) (Amazon Elastic Compute Cloud): *a web service that provides secure, resizable compute capacity in the cloud.*
- [ML Infrastructure with Terraform](https://medium.com/@alexgidiotis_96550/building-ml-infrastructure-with-terraform-520b80874e8b): *open-source Infrastructure as Code (IaC) framework that enables users to define and provision their infrastructure using a high-level configuration language known as HashiCorp Configuration Language (HCL).*
- [VPCs](https://en.wikipedia.org/wiki/Virtual_private_cloud) (Virtual Private Cloud): *on-demand configurable pool of shared resources allocated within a public cloud environment*

### Deploying to Heroku

We will deploy to two repos separately.  First for the backend.

### Preparing the API for deployment

In order to prepare our API for deployment here are some extra tasks.

- make sure debug is off
- add the root route to our API
- add a default JSON renderer for production
- creating a Procfile
- installing a package needed to run the project on Heroku
- creating the Heroku app
- setup and production settings
- fixing a few environment variables

#### Turn off debug

You can simply set debug to false, but if you want to continue to develop locally but have it turned off when deployed, change this:

```py
DEBUG = True
```

to this:

```py
DEBUG = 'DEV' in os.environ
```

#### Add the root route to our API

This is just a simple way to indicate that the app been deployed successfully.

To do this, create a new file in the DRF root directory pytorch_django\views.py:

```py
from rest_framework.decorators import api_view
from rest_framework.response import Response


@api_view()
def root_route(request):
    return Response({
        "message": "You have reached the pytorch-django API"
    })
```

Then the the pytorch_django\urls.py import the root_route and set that as the first default rout.

```py
from django.contrib import admin
from django.urls import path, include
from .views import root_route

urlpatterns = [
    path('', root_route),
    path('admin/', admin.site.urls),
    path('', include('image_classification.urls')),
    path('', include('images.urls')),
```

### Add a default JSON renderer for production

We may only want the in-browser interface to be available in development.  The frontend only uses JSON for this project.
To do this, in settings.py, if the 'DEV' env variable is NOT present, set the rest framework's default renderer  
classes attribute to JSONRenderer inside a list.

It looks like this:

```py
REST_FRAMEWORK = {
    ... 
}
if 'DEV' not in os.environ:
    REST_FRAMEWORK['DEFAULT_RENDERER_CLASSES'] = [
        'rest_framework.renderers.JSONRenderer',
    ]
```

### Create a Procfile

The last to steps were option. This one is required.

Creating a Procfile file will provide commands to Heroku to build and run a Python DRF app.

Remember, it must be named correctly and not have any file extension, otherwise Heroku won’t recognize it

Inside the Procfile, add these two commands

```sh
release: python manage.py makemigrations && python manage.py migrate
web: gunicorn pytorch_django_react.wsgi
```

We will install this gunicorn in a minute.

In settings.py file, update the value of the ALLOWED_HOSTS variable to include your Heroku app’s URL (we will get that next).

```py
ALLOWED_HOSTS = ['localhost', '<your_app_name>.herokuapp.com']
```

#### Installing a package needed to run the project on Heroku

In the terminal of your IDE workspace, install gunicorn

```sh
pip3 install gunicorn django-cors-headers
```

Update your requirements.txt

```sh
pip freeze --local > requirements.txt
```

Add corsheaders to INSTALLED_APPS

```py
INSTALLED_APPS = [
    ...
    'dj_rest_auth.registration',
    'corsheaders',
    ...
 ]
```

Add corsheaders middleware to the TOP of the MIDDLEWARE if you didn't already in a previous step, and also SITE_ID = 1:

```py
 SITE_ID = 1
 MIDDLEWARE = [
     'corsheaders.middleware.CorsMiddleware',
     ...
 ]
```

I'm not actually sure at this point what SITE_ID is for but I will find out and update this when I do.

Set the DEBUG value to be True only if the DEV environment variable exists. This will mean it is True in development, and False in production

```py
DEBUG = 'DEV' in os.environ
```

Comment DEV back in env.py if you want.  I have run without this change, but here is what I have right now:

```py
import os
os.environ['CLIENT_ORIGIN'] = 'http://127.0.0.1:5173'
os.environ['DEV'] = '1'
```

Add, commit and push to GitHub.

### The Stefan Schneider method

There are some differences to the [article deployment section](https://stefanbschneider.github.io/blog/posts/pytorch-django/index.html#deployment-on-heroku).

It's worth pointing as he has a [dedicated blog post](https://stefanbschneider.github.io/blog/posts/django-heroku/index.html) on the topic which might also be useful.  There can always be issues that show up because a detail is missed or something has changed in the way the web host works, or package version, so being able to go more in-depth might be necessary.

Stefan mentions that he had some issues with the file structure.

*For some reason, the default directory structure always breaks my Heroku deployment. It works, when removing the parent pytorch_django directory like this:*

#### original structure when generating the project and app

```txt
pytorch_django
    image_classification
        ...
    pytorch_django
        ...
    manage.py
README.md
```

#### after removing the parent directory

```txt
image_classification
    ...
pytorch_django
    ...
manage.py
README.md
```

However, out of the box this project has the structure shown in the second list.  It might be because we are using an older version of Django.

### Create a Heroku app

The process to create a new app on Heroku is documented on their site.

- log into the Heroku Dashboard
- click "New" and "Create new app"
- name the app and select the region

I didn't know this until I tried it, but pytorch_django will not work as an app name, as the validation warning says "This name should only contain lowercase letters, numbers, and dashes".

So it will have to be pytorch-django.  A bit inconenient, as the GitHub repo is pytorch_django_react, the Django app is actualy just pytorch_django, and now the Heroku app is pytorch-django.

So what does that mean for ths pytorch_django.wsgi.application in WSGI_APPLICATION?  Which one should that be?

The app name in the wsgi.application setting refers to the Python module where the WSGI application object is located, and it doesn't necessarily have to match the exact app name used on the hosting platform (in this case, Heroku).The app name in the wsgi.application setting refers to the Python module where the WSGI application object is located, and it doesn't necessarily have to match the exact app name used on the hosting platform (in this case, Heroku).

So this is the correct setting:

```py
WSGI_APPLICATION = 'pytorch_django.wsgi.application'
```

### Setup and production settings

After creating the app on Heroku and enabling automatic deploys from the corresponding GitHub repo, set the following config variables (in Heroku: Settings > Config Vars):

```txt
DJANGO_SETTINGS_MODULE: pytorch_django.prod_settings
DJANGO_SECRET_KEY: <randomly-generated-secret-key>
```

This indicates that Heroku should use a separate prod_settings.py rather than the settings.py used for development. This prod_settings.py simply overwrites and disables debug mode, sets the production secret key, and allowed hosts. It also makes use of the django_heroku package for further settings.

We are NOT using this place as we have conditional logic in the settings.py file to account for prod vs. dev.

### Deployment errors

I would be surprised if it worked the first time.  No surprises this time:

```err
       Collecting pytz==2023.3.post1 (from -r requirements.txt (line 72))
         Downloading pytz-2023.3.post1-py2.py3-none-any.whl.metadata (22 kB)
       ERROR: Ignored the following versions that require a different python version: 1.21.2 Requires-Python >=3.7,<3.11; 1.21.3 Requires-Python >=3.7,<3.11; 1.21.4 Requires-Python >=3.7,<3.11; 1.21.5 Requires-Python >=3.7,<3.11; 1.21.6 Requires-Python >=3.7,<3.11; 4.4.0 Requires-Python >=3.6,<3.9
       ERROR: Could not find a version that satisfies the requirement pywin32==306 (from versions: none)
       ERROR: No matching distribution found for pywin32==306
 !     Push rejected, failed to compile Python app.
 !     Push failed
```

ChatGPT says: *ince pywin32 is not required for your Django app to function properly, you can remove the pywin32==306 line from your requirements.txt file. Simply delete the line containing pywin32==306, save the file, and attempt the deployment again.*

Then try again.  This turned out to be a very long deployment process.  And again, it fails:

```err
-----> Compressing...
 !     Compiled slug size: 2.6G is too large (max is 500M).
 !     See: http://devcenter.heroku.com/articles/slug-size
 !     Push failed
```

2.6 gigs?  I guess it's the sample images I included.

Delete those and try again with no change.  What's going on?  Then I realize there is no .gitignore file.  How did that happen?  I thought that was a default file.  In my last DRF project I have one, but I don't recall making it.  It looks like this:

```txt
core.Microsoft*
core.mongo*
core.python*
env.py
__pycache__/
*.py[cod]
node_modules/
.github/
cloudinary_python.txt
db.sqlite3
```

```sh
git rm -r --cached .
PS C:\Users\timof\repos\django\pytorch_django_react> git rm -r --cached .
rm '.gitignore'
rm '.vscode/settings.json'
rm 'Procfile'
rm 'README.md'
...

git add .

modified:   README.md
        deleted:    __pycache__/env.cpython-310.pyc
        deleted:    db.sqlite3
        deleted:    env.py
        deleted:    image_classification/__pycache__/__init__.cpython-310.pyc
        ...

git commit -m "remove ignored files"
```

Seemed like that is the right output, but the push still fails.

The static directory is needed, but not those files.

However, I am seeing a lot of messages like this in the log output:

```log
Found another file with the destination path 'admin/js/calendar.js'. It will be ignored since only the first encountered file is collected. If this is not what you want, make sure every static file has a unique path.
```

Why would there be all these duplicates there?

The only file I need in the static dir is this: static\imagenet_class_index.json  why is an admin directory there?

How to find out what is causing such a big bundle size?

Ask ChatGPT
ChatGPT

Poe

To check the size of your project and identify the files contributing to its large size, you can follow these steps:

Install the Heroku CLI if you haven't already. You can download it from the official Heroku website: https://devcenter.heroku.com/articles/heroku-cli

Open a terminal or command prompt and log in to your Heroku account by running the following command:

Copy
heroku login
Navigate to the root directory of your project in the terminal.

Use the Heroku CLI to create a new Heroku app (if you haven't created one already) by running:

Copy
heroku create
Deploy your project to Heroku by pushing it to the Heroku Git remote:

Copy
git push heroku main
During the deployment process, you will see the output that includes the line "Compiled slug size: X is too large (max is 500M)." This indicates that the compiled slug size exceeds the maximum allowed size of 500MB.

After the deployment fails, you can check the size of your project's slug by running the following command:

Copy
heroku run du -sh
This command will display the total size of your project in a human-readable format.

To investigate further and identify the files contributing to the large slug size, you can run the following command to list the files and their sizes within the slug:

Copy
heroku run bash -c "du -h -d 1"
This command will provide a directory-wise breakdown of file sizes within the slug.

By following these steps, you can determine the size of your project and identify the files that are contributing to the large slug size, helping you optimize your project and reduce its size to meet the maximum allowed limit.


