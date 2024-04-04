# Deploying to Heroku

These are the steps followed in an attempt to use Heroku for the backend app.

## Legacy notes

Long story short, the failure was due to "Compiled slug size: 2.6G is too large (max is 500M)."

Regardless, below are an archive of those notes.

### Create a Heroku app

The process to create a new app on Heroku is documented on their site.

- log into the Heroku Dashboard
- click "New" and "Create new app"
- name the app and select the region

I didn't know this until I tried it, but pytorch_django will not work as an app name, as the validation warning says "This name should only contain lowercase letters, numbers, and dashes".

So it will have to be pytorch-django.  A bit inconvenient, as the GitHub repo is pytorch_django_react, the Django app is actualy just pytorch_django, and now the Heroku app is pytorch-django.

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

I notice that there are two directories, on static, and one staticfiles:

```py
STATIC_URL = '/static/'

STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static'),]
```

When I run the app, the staticfiles directory gets created with an admin directory in it.  Is this the problem?  Stefan has the exact code in his article.

when deploying your Django project to Heroku, you can specify the --ignore flag as a configuration option in your Procfile.

So doing this in the Procfile helps to resolve these warnings:

```py
release: python manage.py makemigrations && python manage.py migrate --noinput --ignore=admin
```

However, the bundle size is still huge.  Looking at the five biggest packages listed in the log, I can see the issue:

```log
torch-2.2.1-cp312-cp312-manylinux1_x86_64.whl (755.5 MB)
nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl (731.7 MB)
torchvision-0.17.1-cp312-cp312-manylinux1_x86_64.whl (6.9 MB)
numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.0 MB)
pandas-2.2.1-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.7 MB)
```

If Heroku has a max deployment size of 500M, does that mean that it's not possible to deploy this app?  It might have been possible in early 2021, but not anymore.

What are the options then?

1. Amazon Web Services (AWS): AWS provides a wide range of services, including EC2 instances and Elastic Beanstalk, which can handle larger deployments and offer more flexibility in terms of storage and resources.
2. Google Cloud Platform (GCP): GCP offers services such as Compute Engine and App Engine, which provide scalable infrastructure for hosting applications with larger file sizes.
3. Microsoft Azure: Azure provides options like Azure App Service and Azure Virtual Machines, which can handle larger deployments and offer various storage and compute options.

I would choose either 1 or 3.  Given the size of AWS and it's economies of scale, that's probably the best option, career-wise.
