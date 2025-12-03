# Kujenga

The building blocks for learning Data Science, AI and Computational Thinking. This is the intro course in our African AI series. It is the source code for the Kujenga webpage, but can also be used as a template for other courses.

# Installation

## Anaconda

To install the required packages use Conda and the following command will make an environment called course.
```
conda env create -f environment.yml
```
## Without Anaconda

in the project home directory , create a virtual environment with the following command:
```
python -m venv venv
```
Activate the environment with:
- On Windows:

```
 venv/Scripts/activate
```
- On MacOS/Linux:
    source venv/bin/activate

Then install the required packages with:

```
pip install -r course/requirements-pip.txt
```

# Create the docs locally

To create the docs run the following commands from the 'course' directory:

```python
make clean
make html
```

You will find the course webpage in the directory build/html/index.html


# How to make changes

There are primarily two types of files the make up the course *source* and *lessons* . here we describe how to work with them.

## Source

These are .rst (restructured text) files stored in the lessons folder. Here is an intro to restructured text https://www.writethedocs.org/guide/writing/reStructuredText/. It is also possible to use markdown format, but you are encouraged to use .rst so we can easily reuse formetting etc.

## Lessons

These are .py files that run python code. The name of these files should always begin with 'plot_'. The advantage of these files is that they are compile into webpages, python code and into a python notebook. You just need to write python code, with all the explanation text commented out. See some of the files for examples. But basically, you can use """ at the start to enclose the introductory text and then a line of '##############' to start sections within this. This dhoul be written in .rst format.

*Important:* There needs to be a README.rst file in every subdirectory of the lessons folder for some reason or else these won't work.

