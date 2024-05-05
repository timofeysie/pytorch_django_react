# CORS

[Cross-Origin Resource Sharing](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS) allows a server to indicate origins than its own from which a browser should permit loading resources.

In this case, we have the DRF API running on http://3.26.241.7:8000/

The frontend is deployed separately on https://pytorch-frontend.vercel.app/, or locally on http://127.0.0.1:5173/ which is the default that the Vite created frontend uses.

This means that requests from the frontend will be blocked unless things are setup correctly to allow CORS.

Since this always results in a few failed attempts, having a clear checklist to go through to debug a setup is the goal of this article.

## CORS Run through

In my Django project's settings.py file, I have 'corsheaders' to the INSTALLED_APPS list and set the CORS_ALLOWED_ORIGINS like this:

```py
INSTALLED_APPS = [
    'image_classification.apps.ImageClassificationConfig',
    'corsheaders',
    'django.contrib.admin',
    ...
]
SITE_ID = 1
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    ...
]

if 'CLIENT_ORIGIN' in os.environ:
    CORS_ALLOWED_ORIGINS = [
        os.environ.get('CLIENT_ORIGIN')
    ]
else:
    CORS_ALLOWED_ORIGINS = [
        'http://localhost:3000',
        'https://pytorch-frontend.vercel.app/',
    ]
```

The output when the app is run shows:
['http://127.0.0.1:5173']

The client origin could be held in the env.py file under CLIENT_ORIGIN, or set in the else statement which has hardwired values.

In env.py I have this for running locally:

```py
os.environ['CLIENT_ORIGIN'] = 'http://127.0.0.1:5173' # local
```
