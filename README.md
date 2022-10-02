ml_template
==============================

Instructions
------------

1. Run `make venv` to create a python venv named "myvenv" with the same python version as your default binary `/usr/bin/python3`.
An easy way of changing your python version is to use miniconda to first create a conda environment with python 3.x and then run `make venv` after activating that conda environment. The advantage of venv is that now your python environment files are stored in this directory (but not committed to git).
2. Run `make pre-commit` to make ensure that we use `black` to format our code before every commit
3. Activate the venv by `source ./myvenv/bin/activate` and run the python scripts!

About pre-commit
----------------

After installing `pre-commit`, it will run the git hooks every time you do `git commit` to check that the files are properly formatted. If it fails, you just need to fix the issues. If you have `black` and `isort` installed, simply `git add` and `git commit` again and the hooks will automatically run `black` and `isort`. See [here](https://interrupt.memfault.com/blog/pre-commit) for more information about pre-commit.