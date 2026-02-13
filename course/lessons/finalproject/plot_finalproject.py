"""

.. _project:

Final Project
=============

Your Mission
------------
Congratulations for making it this far :)! You've learned some powerful mathematical and computational tools. Now it's time to put them to work on something you care about.

Your final project is straightforward: **Pick a real-world question, find data to answer it, and use the techniques you've learned to tell a compelling story.**

.. youtube:: QQ0YYR2BhFk
    :width: 100% 
    :align: center

In the video above, we walk through what makes a great final project and show you examples from past students who have successfully applied what they've learned to answer questions they cared about.

What Makes a Great Project?
----------------------------
The best projects have three ingredients:
1. **A Question You Actually Care About** 
   
   * "Are electric vehicles becoming more popular in my country?"
   * "Does coffee consumption correlate with productivity?"
   * "Can I predict which football team will win based on past performance?"
   * "How has air quality changed in major cities over the past decade?"

2. **Data You Can Get Your Hands On**
   
   * More on this below - but don't worry, we'll show you where to look!

3. **At Least One Technique From This Course**
   
   * Linear regression (like the happiness-health analysis)
   * Statistical testing (like the Ethiopian vs Kenyan runners)
   * Differential equations (like the SIR model)
   * Network analysis (like PageRank)

Project Ideas to Get You Started
---------------------------------
Not sure where to begin? Here are some ideas:

**Health & Lifestyle**
* Does screen time affect sleep quality in students?
* Is there a relationship between exercise frequency and mental health?
* Do vaccination rates correlate with disease incidence?

**Economics & Social Issues**
* How do education levels relate to income in different countries?
* Is there a gender pay gap in certain industries?
* Do crime rates correlate with unemployment?

**Environment & Climate**
* How has temperature changed in your city over the past 50 years?
*Can you predict flood risk based on rainfall, elevation, and river discharge data? 
* Does air pollution correlate with respiratory diseases?
* Can you model population growth in your region?

**Sports & Entertainment**
* What factors predict a team's success in your favorite sport?
* Do movie budgets correlate with box office success?
* Can you model the spread of a viral trend on social media?
*Do NFL kickers make more field goals when the crowd is louder? 

**Technology & Innovation**
* How has internet access grown globally?
* Is there a relationship between smartphone adoption and economic development?
* Can you predict tech stock prices using trends?

Where to Find Data
------------------
The internet is full of free, accessible datasets! Here are the best places to start:

**General Data Repositories**

* `Kaggle Datasets <https://www.kaggle.com/datasets>`_ - Thousands of clean, ready-to-use datasets
* `Google Dataset Search <https://datasetsearch.research.google.com/>`_ - Like Google but for data
* `Our World in Data <https://ourworldindata.org/>`_ - Beautiful, clean data on global issues
* `Data.gov <https://data.gov/>`_ - US government open data
* `UCI Machine Learning Repository <https://archive.ics.uci.edu/>`_ - Classic datasets

**Specific Topics**

* **Health**: `WHO Data <https://www.who.int/data>`_, `CDC Data <https://data.cdc.gov/>`_
* **Economics**: `World Bank <https://data.worldbank.org/>`_, `IMF Data <https://www.imf.org/en/Data>`_
* **Sports**: `Sports Reference <https://www.sports-reference.com/>`_, `Kaggle Sports <https://www.kaggle.com/search?q=sports+in%3Adatasets>`_
* **Climate**: `NASA Climate Data <https://climate.nasa.gov/vital-signs/carbon-dioxide/>`_, `NOAA <https://www.noaa.gov/>`_
* **Social Media**: `Twitter API <https://developer.twitter.com/>`_, `Reddit API <https://www.reddit.com/dev/api/>`_

**Tip**: Start with Kaggle or Our World in Data - they have clean, well-documented datasets that work well for learning.

Cleaning Your Data
------------------
Real-world data is messy. Here's how to clean it:

"""

from markdown import Markdown
import pandas as pd

##############################################################################
# Step 1: Load and Inspect
# ^^^^^^^^^^^^^^^^^^^^^^^^
# First, load your data and take a look at what you're working with.
# This helps you understand the structure and identify potential issues.

# Load your data

#data = pd.read_csv('your_data.csv')

# Take a look
#print(data.head())
#print(data.info())
#print(data.describe())

##############################################################################
# Step 2: Handle Missing Values
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Missing data is common. You can either remove rows with missing values
# or fill them with appropriate values like the mean or median.

# See where data is missing
#print(data.isnull().sum())

# Option 1: Drop rows with missing values
#data_clean = data.dropna()

# Option 2: Fill missing values with mean/median
#data['column_name'].fillna(data['column_name'].mean(), inplace=True)

##############################################################################
# Step 3: Remove Duplicates
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# Duplicate rows can skew your analysis. Check for and remove them.

# Check for duplicates
#print(data.duplicated().sum())

# Remove them

#data_clean = data.drop_duplicates()

##############################################################################
# Step 4: Fix Data Types
# ^^^^^^^^^^^^^^^^^^^^^^^
# Make sure each column has the correct data type. Dates should be datetime
# objects, numbers should be numeric types, etc.

# Convert to the right type

#data['date_column'] = pd.to_datetime(data['date_column'])
#data['numeric_column'] = pd.to_numeric(data['numeric_column'])

##############################################################################
# Step 5: Filter Outliers (if needed)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Extreme values can distort your results. Use the interquartile range (IQR)
# method to identify and remove outliers.

# Remove extreme values

#Q1 = data['column'].quantile(0.25)
#Q3 = data['column'].quantile(0.75)
#IQR = Q3 - Q1

#data_clean = data[(data['column'] >= Q1 - 1.5*IQR) & 
#                  (data['column'] <= Q3 + 1.5*IQR)]

##############################################################################
# Need more help? Check out the `Pandas documentation <https://pandas.pydata.org/docs/>`_ or this `data cleaning tutorial <https://realpython.com/python-data-cleaning-numpy-pandas/>`_.
#
#Project Requirements
#--------------------
#
#Your final project should include:
#
#1. **A Clear Question** - What are you trying to find out?
#
#2. **Data Description** - Where did you get your data? What does it contain?
#
#3. **Data Cleaning** - Show how you cleaned and prepared your data
#
#4. **Analysis** - Use at least one technique from the course (regression, t-test, modeling, etc.) # type: ignore
#
#5. **Visualization** - Create at least 2 plots that help tell your story
#
#6. **Conclusion** - What did you discover? What are the limitations?
#
#7. **Code** - Submit your Jupyter notebook or Python script
#
#Submission Format
#-----------------
#Depending on your instructor's requirements, you may submit either a Jupyter notebook or a Python script. However, we highly recommend using a Jupyter notebook for its ability to combine code, visualizations, and narrative in one place.
#Submit a Jupyter notebook that includes:
#* Markdown cells explaining your thinking
#* Code cells showing your analysis
#* Visualizations
#* A final conclusion section
#
#**File naming**: ``yourname_final_project.ipynb``
#
#
#Tips for Success
#----------------
#
# * **Start simple** - Better to do one thing well than many things poorly
# * **Tell a story** - Guide your reader through your thinking
# * **Make it visual** - Good plots make your findings memorable
# * **Be honest** - If your hypothesis was wrong, that's okay! Explain what you learned
# * **Ask for help** - Stuck? Reach out to your instructor or classmates
#
#
#Getting Started
#---------------
#
# 1. Pick a question that excites you
# 2. Find a dataset (start with Kaggle or Our World in Data)
# 3. Download the data and start exploring
# 4. Clean it up and apply what you've learned
# 5. Create visualizations that tell your story
# 6. Share your findings
# 7. Celebrate your hard work and new skills!
#
#Remember: The best projects are the ones where you learn something new about a topic you care about.
#
#Have Any Questions? Ask your instructor.