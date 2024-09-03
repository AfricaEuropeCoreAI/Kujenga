#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 10:43:45 2023

@author: davsu428
"""


import numpy as np
import pandas as pd
import pyreadstat
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import random

from pylab import rcParams
rcParams['figure.figsize'] = 12/2.54, 12/2.54

# This function rescales the data slightly to help do the analysis.
def Rescale(value, scale_spending):
    if value>=0:
        Q = scale_spending[int(value)]
    else:
        Q =np.nan
    return Q


# To use this you have to have access to the original data set
# from [LINK HERE]. On request from authors.
df, meta = pyreadstat.read_sav("../data/Study7_QualtricsData.sav")

# The data is coded for different spending levels. 
# This converts the spending levels in to dollars.
scale_spending = np.array([0] + list(range(10,100,20)) + list(range(150,1000,100)) + [1000] + [1000])
scale_spending = np.append(scale_spending, [np.nan,np.nan,np.nan])
scale_spending_labels = meta.value_labels['labels7']

# There are three different happiness measures in the data
# using different measures gives only slightly different results. 

happiness_measure='SWL2'

for original_name in ['MATA','Bills','TSA','EXPA']:
    df['spent_on_' + original_name] = df.apply(lambda row: Rescale(row[original_name],scale_spending), axis=1)


df=df.rename({happiness_measure: 'Happiness'}, axis=1)

df=df.drop(['Respondents','SWL1','TP1',
 'TP1',
 'TP2',
 'TP3','MATA','Bills','TSA','EXPA',
 'EXP',
 'MAT',
 'TS',
 'Items',
 'Kids',
 'Marital',
 'Employment',
 'Age',
 'Hours',
 'Inc',
 'Ethnicity',
 'Gen',
 'XX_Composites_XX',
 'SWL',
 'TP',
 'TP_C',
 'TPXTS',
 'SurveyComments',
 'Completed',
 'filter_$',
 'Inc_Log',
 'LINCXTS',
 'Age_C',
 'Age_Sq',
 'Inc_C',
 'INCXTS',
 'AMT_C',
 'AMT_SQ'], axis=1)



df.to_csv('../data/spending.csv')