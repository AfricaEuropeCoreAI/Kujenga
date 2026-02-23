"""

.. _project:

Final Project
=============

Your Mission
------------
Congratulations for making it this far :)! You've learned some powerful mathematical and computational tools. Now it's time to put them to work on something you care about.

Your final project is straightforward: **Pick a real-world question, find data to answer it, and use the techniques you've learned to tell a compelling story.**

[VIDEO COMING SOON HERE]

In the video above, we walk through what makes a great final project and show you examples from past students who have successfully applied what they've learned to answer questions they cared about.

What Makes a Great Project?
----------------------------
The best projects have three ingredients:

1. **A Question You Actually Care About** 
 It doesn't have to be earth-shattering. It just has to be something that matters to you and your community.
 The more specific and local, the better. The best questions are the ones you see in your daily life.  
2. **Data You Can Get Your Hands On**
 More on this below - but don't worry, we'll show you where to look!
3. **At Least One Technique From This Course**
   
   * Linear regression (like the happiness-health analysis)
   * Statistical testing (like the Ethiopian vs Kenyan runners)
   * Differential equations (like the SIR model)
   * Network analysis (like PageRank)

Project Ideas to Get You Started
---------------------------------
Not sure where to begin? Start with a question that genuinely matters to you and your community. The most powerful machine learning projects are not the ones that sound impressive, they are the ones that solve real problems close to home.
For this course, we encourage you to focus on **African challenges and local realities**. Think about the issues you see every day. Flood prediction in informal settlements. Crop disease detection for smallholder farmers. 
Electricity outage patterns in your county. Traffic congestion in your city. Air quality near busy roads. Youth unemployment trends in your area.

For example, let’s say you’re from Kenya and you’ve noticed how loud matatu music can be during long commutes. You might ask: Is there a correlation between prolonged exposure to high-decibel matatu music and hearing issues among commuters? That question can turn into a real machine learning project — collecting survey data, measuring exposure time, analysing patterns, and testing whether there is correlation (and carefully exploring whether causation might be possible to infer).

This is exactly the spirit of the Kujenga course:
Start with what you see.
Ask a meaningful question.
Use data to investigate it.
Your project does not have to be perfect. It just has to be relevant and thoughtful.

Where to Find Data
------------------
The internet is full of free, accessible datasets, but don’t limit yourself to global sources only. Some of the most impactful projects come from local African datasets.
You can explore:

* Government open data portals (for example, national statistics offices, health ministries, meteorological departments).
* African research institutions and NGOs.
* County-level public records.
* International datasets with African coverage (World Bank, UN data, etc.).
* Community-collected surveys and field data.

And remember: you are not restricted to pre-existing datasets. You can collect your own data through surveys, simple measurements, interviews, or partnerships with local organisations.
If you ever feel stuck, these repositories below can be your starting point.
The goal is to find data that helps you answer a question you genuinely care about.
**General Data Repositories**

* `Kaggle Datasets <https://www.kaggle.com/datasets>`_ - Thousands of clean, ready-to-use datasets
* `Google Dataset Search <https://datasetsearch.research.google.com/>`_ - Like Google but for data
* `Our World in Data <https://ourworldindata.org/>`_ - Beautiful, clean data on global issues
* `Data.gov <https://data.gov/>`_ - US government open data
* `UCI Machine Learning Repository <https://archive.ics.uci.edu/>`_ - Classic datasets

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
#**File naming**: ``yourname_yourcountry_final_project.ipynb``
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