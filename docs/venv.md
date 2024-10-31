# virtual environments for Python

While learning how to Securely Deploy a Django App With ]Gunicorn, Nginx, & HTTPS](https://realpython.com/django-nginx-gunicorn/) I went down the rabbit hole of [virtual envs](https://realpython.com/python-virtual-environments-a-primer/)

The primer version shows these commands:

```sh
PS> python -m venv venv
PS> venv\Scripts\activate
… install with pip
(venv) PS> deactivate
PS>
```

For DNG version (for Python 3.8.10 Ubuntu):

```sh
$ python3 -m venv env
$ source env/bin/activate
```

## On Windows

Using VSCode with a powershell terminal:

```sh
.\venv\Scripts\activate
(venv) PS > pip install xyz
(venv) PS > deactivate
pip freeze --local > requirements.txt
```
