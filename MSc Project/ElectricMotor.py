#!/usr/bin/env python
# coding: utf-8

# # MSc Project. David Brookes, June-August 2020.
# 

# In[1]:


# Import libraries.
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import scipy

import joblib


# In[2]:


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"


# In[3]:


# To plot figures.
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# In[4]:


# Where to save the images.
PROJECT_ROOT_DIR = "."

def save_fig(IMAGE_Subfolder, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "Images", IMAGE_Subfolder)
    os.makedirs(IMAGES_PATH, exist_ok=True)
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# In[6]:


# Where to save the models.
PROJECT_ROOT_DIR = "."
#MODELS_PATH = os.path.join(PROJECT_ROOT_DIR, "Models")

#os.makedirs(MODELS_PATH, exist_ok=True)

def save_model_and_results(model, results, MODEL_Subfolder, filename, extension="pkl"):
    MODELS_PATH = os.path.join(PROJECT_ROOT_DIR, "Models", MODEL_Subfolder)
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    model_filename = filename + '+' + MODEL_Subfolder
    path = os.path.join(MODELS_PATH, model_filename + "." + extension)
    print("Saving model", model_filename)
    joblib.dump(model, path)
    
    results_filename = filename + '+' + 'results'
    path = os.path.join(MODELS_PATH, results_filename + "." + extension)
    print("Saving results", results_filename)
    joblib.dump(results, path)

def save_all_models_and_results(best_est_dict, cv_metrics_dict, MODEL_Subfolder):
    for y_var in response_variables:
        best_est = best_est_dict[y_var]
        cv_metrics = cv_metrics_dict[y_var]
        filename = y_var
        save_model_and_results(best_est, cv_metrics, MODEL_Subfolder, filename) 

    
def load_model_and_results(MODELS_Subfolder, filename, extension="pkl"):
    MODELS_PATH = os.path.join(PROJECT_ROOT_DIR, "Models", MODEL_Subfolder)  
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    model_filename = filename + '+' + MODEL_Subfolder
    path = os.path.join(MODELS_PATH, model_filename + "." + extension)
    model_loaded = joblib.load(path)
    
    results_filename = filename + '+' + 'results'
    path = os.path.join(MODELS_PATH, results_filename + "." + extension)
    results_loaded = joblib.load(path)
    
    return(model_loaded, results_loaded)
    
def load_all_models_and_results(MODEL_Subfolder):
    best_estimators = []
    cv_metrics_all = []
    for y_var in response_variables:
        best_est, cv_metrics = load_model_and_results(MODEL_Subfolder, y_var)
        best_estimators.append(best_est)
        cv_metrics_all.append(cv_metrics)
    return(best_estimators, cv_metrics_all)

      


# In[7]:


# Print current working directory (folder).
curr_dir=os.getcwd()
print(curr_dir)


# In[8]:


# Change working directory.
# Note: 'r' allows backslashes (and forward slashes) in the file path name.
#curr_dir = os.chdir(r"D:\My Documents\Essex University\MSc Project\Electric Motor\Data set")
curr_dir = os.chdir(r"D:\My Documents\Essex University\MSc Project\Electric Motor")


# In[8]:


# List the contents of the working directory.
#os.listdir(curr_dir)


# In[9]:


# Read in the electric motor data. Store as a dataframe.
#
# Note: 'r' allows backslashes (and forward slashes) in the file path name.
motor_data = pd.read_csv(r"Data set\pmsm_temperature_data.csv")


# In[10]:


# Use a sample of the data for testing purposes to save time.
# Set  FRAC to 1.0 to use all the data.
FRAC = 1.0


# In[24]:


# Determine if there are any missing values.
motor_data.isnull().values.any()


# In[25]:


# Count the number of NaNs each column has.
nans=pd.isnull(motor_data).sum()
nans[nans>0]


# In[26]:


# Count the column types.
motor_data.dtypes.value_counts()


# In[11]:


# Print out the number of rows and columns in the dataframe.
motor_data.shape


# In[28]:


# Print the first five rows of the data (head()).
motor_data.head()


# # Session Lengths.

# In[29]:


# Calculate the session lengths and sort into ascending order.

# Where to save the session lengths.
IMAGE_Subfolder = "Session lengths"

df=motor_data
fig = plt.figure(figsize=(17, 5))
grpd = df.groupby(['profile_id'])
_df = grpd.size().sort_values().rename('samples').reset_index()
ordered_ids = _df.profile_id.values.tolist()
sns.barplot(y='samples', x='profile_id', data=_df, order=ordered_ids)
tcks = plt.yticks(2*3600*np.arange(1, 8), [f'{a} hrs' for a in range(1, 8)]) # 2Hz sample rate
save_fig(IMAGE_Subfolder, "session_lengths")


# In[17]:


# Print information on the motor data.
#motor_data.info()


# In[12]:


# Drop the profile_id variable. This identifies the measurement sessions.
#
# Note: Carefully distinguishing between the profiles is only relevant if 
# training a model with memory capabilities, e.g. LSTM/GRU, or if using
# lag-features (e.g. implicitly with CNNs).

auxiliary_variables = ['profile_id']
drop_cols = auxiliary_variables
motor_data = motor_data.drop(drop_cols, axis=1)

#motor_data.info()


# In[31]:


# Drop duplicate rows.
motor_data = motor_data.drop_duplicates(keep='first')
#motor_data.info()


# In[13]:


# Print out the number of rows and columns in the dataframe.
motor_data.shape


# In[21]:


# Print summary statistics
# The data has been normalised so that mean=0 and standard deviation=1 approximately.

motor_data.describe().T


# In[14]:


# Assign more meaningful variable names to a list object.
column_fullnames = ['Ambient Temperature','Coolant Temperature','Voltage d-component','Voltage q-component','Motor Speed','Torque','Current d-component','Current q-component','Permanent Magnet Surface Temperature','Stator Yoke Temperature','Stator Tooth Temperature','Stator Winding Temperature']

column_shortnames = list(motor_data.columns)
print('column_shortnames', column_shortnames)

# This is a list of response (target) variables that need to be predicted.
response_variables = ['torque','pm','stator_yoke','stator_tooth','stator_winding']
print('response_variables', response_variables)

# Remember : auxiliary_variables = ['profile_id']

# Form a list of predictor (input) variables.
def generate_pred_vars(response_variables, auxiliary_variables, column_names):
    predictor_variables = column_names.copy()
    for el in response_variables+auxiliary_variables:
        if el in column_names:
            predictor_variables.remove(el)
    return(predictor_variables)

predictor_variables = generate_pred_vars(response_variables, auxiliary_variables, column_shortnames)

print('predictor_variables', predictor_variables)


# # Histograms.

# In[23]:


# Where to save the histograms.
IMAGE_Subfolder = "Histograms"

# Plot histogram of the variables, and also the kernel density estimate (KDE).
# Note: KDE is an estimate of the probabilty density function (pdf).

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use the ALL data.
sample_motor_data = motor_data.sample(frac = FRAC)

num_bins = 50
for i in range(0, len(sample_motor_data.columns)):
    
    sns.distplot(sample_motor_data[column_shortnames[i]], bins=num_bins, axlabel=column_fullnames[i],kde_kws={"color": "k", "linewidth": 3, "label": "KDE", "color": "b"},hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "g"})
   
    plt.xlabel(column_fullnames[i])
    plt.ylabel('Frequency Density')
    plt.title('Histogram of '+ column_fullnames[i])
    
    save_fig(IMAGE_Subfolder, column_shortnames[i])
    
    plt.show()


# # Quantile-Quantile plots.

# In[24]:


# Quantile-Quantile plots for the variables.
# The standard normal distribtion is the comparison distribution.

# Where to save the QQ plots.
IMAGE_Subfolder = "QQ"

import numpy as np
import statsmodels.api as sm
import pylab

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use the ALL data.
sample_motor_data = motor_data.sample(frac = FRAC)

for var in column_shortnames:
    index = column_shortnames.index(var)
    print(column_fullnames[index],'(',var,')')
    #sm.qqplot(motor_data[var], line='45')
    sm.qqplot(sample_motor_data[var], line='s')
    save_fig(IMAGE_Subfolder, var)
    pylab.show()


# # Correlation Matrix.

# In[81]:


# Produce a correlation matrix.

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use the ALL data.
sample_motor_data = motor_data.sample(frac = FRAC)

sample_motor_data_corr = sample_motor_data.corr()
#print(sample_motor_data_corr)


# # Correlation Map.

# In[26]:


# Correlation map.

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use the ALL data.
sample_motor_data = motor_data.sample(frac = FRAC)

# Where to save the correlation map.
IMAGE_Subfolder = "Correlation"

f,ax=plt.subplots(figsize=(12,12))

sample_motor_data_corr=sample_motor_data.corr()

sns.heatmap(sample_motor_data_corr, annot=True, linewidths=.5, fmt='.2f', 
            mask= np.zeros_like(sample_motor_data_corr,dtype=np.bool), 
            cmap=sns.diverging_palette(100,200,as_cmap=True), 
            square=True, ax=ax)
save_fig(IMAGE_Subfolder, "correlation_map")

plt.show()


# # Boxplots All.

# In[27]:


# Print boxplots of variables.

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use the ALL data.
sample_motor_data = motor_data.sample(frac = FRAC)

# Where to save the boxplots.
IMAGE_Subfolder = "Boxplots All"

box_plot_motors = sns.boxplot(data=sample_motor_data, orient='h')
save_fig(IMAGE_Subfolder, "boxplots_all")


# # Boxplots Individual.

# In[28]:


# Print boxplots of variables.

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use the ALL data.
sample_motor_data = motor_data.sample(frac = FRAC)

# Where to save the boxplots.
IMAGE_Subfolder = "Boxplots Individual"

for i in range(0, len(sample_motor_data.columns)):
    sns.boxplot(x=sample_motor_data.columns[i],data=sample_motor_data, orient='h')
    save_fig(IMAGE_Subfolder, sample_motor_data.columns[i])
    plt.show()
 


# # Simple Linear Regression.

# In[29]:


# Simple linear regression.  (Using statsmodels.api).
# 5 models considered.
# Response variables:
# 'torque' Torque induced by current.
# 'pm' (Permanent Magnet surface temperature - representing the rotor temperature).
# 'stator_yoke' (Stator yoke temperature measured with a thermal sensor)
# 'stator_tooth' (Stator tooth temperature measured with a thermal sensor)
# 'stator_winding' (Stator winding temperature measured with a thermal sensor)

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import f
import statsmodels.api as sm # Loaded in order to calculate the starndard error of the estimators (coefficients).
from sklearn.model_selection import KFold # Used for K fold cross validation.
from statistics import mean, variance, stdev 

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use the ALL data.
sample_motor_data = motor_data.sample(frac = FRAC)

for y_var in response_variables:
    
    y = sample_motor_data[y_var]
    #print(y)
    
    for x_var in predictor_variables:
        print('Response variable:  ', y_var)
        print('Predictor variable: ', x_var)
        
        x = sample_motor_data[x_var]
        # Create design matrix x_1.
        x_1 = sm.add_constant(x) # Add a column of 1s.

   
        # Fit a linear model and display the results.
        lm=sm.OLS(y, x_1)
        model=lm.fit()
        print(model.summary())  
        print(' ')
    


# In[30]:


# Print simple linear regression plots of response variables against predictor variables.
# 5 models considered.
# Response variables:
# 'torque' Torque induced by current.
# 'pm' (Permanent Magnet surface temperature - representing the rotor temperature).
# 'stator_yoke' (Stator yoke temperature measured with a thermal sensor)
# 'stator_tooth' (Stator tooth temperature measured with a thermal sensor)
# 'stator_winding' (Stator winding temperature measured with a thermal sensor)   

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use the ALL data.
sample_motor_data = motor_data.sample(frac = FRAC)

# Where to save the simple regression plots.
IMAGE_Subfolder = "Simple regression plots"

for y_var in response_variables:
    for x_var in predictor_variables:
        print('Response variable:  ', y_var)
        print('Predictor variable: ', x_var)
        
        #sns.regplot(x=sample_motor_data.iloc[:,i], y=sample_motor_data['pm'], data=sample_motor_data,
        #             scatter_kws={"color": "black"}, line_kws={"color": "red"})
        sns.regplot(x=sample_motor_data[x_var], y=sample_motor_data[y_var], data=sample_motor_data,
                     scatter_kws={"color": "blue", "s":1, "alpha" : 0.002}, line_kws={"color": "red"})
        save_fig(IMAGE_Subfolder, y_var + '_vs_' + x_var)
        plt.show()
    


# # Multiple Linear Regression.

# In[31]:


# Multiple linear regression. (Using statsmodels.api).
# 5 models considered.
# Response variables:
# 'torque' Torque induced by current
# 'pm' (Permanent Magnet surface temperature - representing the rotor temperature).
# 'stator_yoke' (Stator yoke temperature measured with a thermal sensor)
# 'stator_tooth' (Stator tooth temperature measured with a thermal sensor)
# 'stator_winding' (Stator winding temperature measured with a thermal sensor)

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import f
import statsmodels.api as sm # Loaded in order to calculate the starndard error of the estimators (coefficients).
from sklearn.model_selection import KFold # Used for K fold cross validation.
from statistics import mean, variance, stdev 

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use the ALL data.
sample_motor_data = motor_data.sample(frac = FRAC)

kf = KFold(n_splits=5, shuffle=True, random_state=1)

response_variables = ['torque','pm','stator_yoke','stator_tooth','stator_winding']

rows, cols  =  sample_motor_data.shape

N = rows # The number of observations.
p = len(predictor_variables) # The number of predictor variables.

X = sample_motor_data[predictor_variables]
#X = X.to_numpy()
# Create design matrix X.
X = sm.add_constant(X) # Add a column of 1s.

for y_var in response_variables:
    
    print('Response variable: ', y_var)
    print('Predictor variables: ')
    print(predictor_variables) # Print out the predictor variables.
    
    y = sample_motor_data[y_var]
    #print(y)
   
    # Fit a multilinear model and display the results for ALL the data.
    print('Summary of multilinear model using ALL data: ')
    lm=sm.OLS(y, X)
    model=lm.fit()
    print(model.summary())  
    print(' ')
    
    # Produce a residual plot using ALL the data.
    y_pred = model.predict(X)
    resid = y-y_pred
    colors = 'b'
    area = 1
    plt.scatter(x=y, y=resid, s=area, c=colors, alpha=0.002)
    plt.title('Residual Plot')
    plt.xlabel(y_var)
    plt.ylabel('residual')
    plt.show()
    print(' ')
    
       
    R2_scores = []
    adjusted_R2_scores = []
    rse_scores = []
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index) 

        #X_train, X_test = X[train_index], X[test_index]
        #y_train, y_test = y[train_index], y[test_index]
        
        X_train, X_test = X.iloc[list(train_index)], X.iloc[list(test_index)]
        y_train, y_test = y.iloc[list(train_index)], y.iloc[list(test_index)]
    
        lm=sm.OLS(y_train, X_train)

        # Fit a multilinear model using the training data.
        model=lm.fit()
        #print(model.summary())
        
        # Predict the response for the test data.
        y_pred = model.predict(X_test)
        
        # Calculate metrics on the test data.
        # The coefficient of determination: 1 is perfect prediction. Use test data.
        R2_score = r2_score(y_test, y_pred)
        R2_scores.append(R2_score)
        
        adjusted_R2_score = 1 - (1-R2_score)*(N-1)/(N-p-1)
        adjusted_R2_scores.append(adjusted_R2_score)        
        
        # The residual standard error = estimate of the standard deviation of the residuals.
        mse = mean_squared_error(y_test, y_pred)
        rse_score = math.sqrt(mse)
        rse_scores.append(rse_score)
        
    print('Summary of multilinear model using the TEST data: ')
    mean_R2_score = mean(R2_scores)
    print('R2 score: ', mean_R2_score)
    
    mean_adjusted_R2_score = mean(adjusted_R2_scores)        
    print('Adjusted R2 score: ', mean_adjusted_R2_score)  
    
    mean_rse_score = mean(rse_scores)
    print('RSE score: ', mean_rse_score)
    print(' ')
 


# USING ALL DATA:-
# Note there are 5 linear models being considered. 
# The response variables being; 'torque','pm','stator_yoke','stator_tooth' and 'stator_winding' respectively.
# Null Hypothesis: All coefficients of the model are zero.
# Since p_value = 0 given the F statistic value, this means that the Null Hypothesis can be rejected in all 5 models.
# The p values imply that NOT ALL of the coefficients of the linear model are zero.
# 
# 
# Response variable: 'torque'.
# All p values associated with t statistics of the predictor variables are zero, suggesting that all
# predictor variables are related to 'torque'.
# 
# Response variable: 'pm'.
# All p values associated with t statistics of the predictor variables are zero, suggesting that all
# predictor variables are related to 'pm'.
# 
# Response variable: 'stator_yoke'.
# All p values associated with t statistics of the predictor variables are zero, suggesting that all
# predictor variables are related to 'stator_yoke'.
# 
# Response variable: 'stator_tooth'.
# All p values associated with t statistics of the predictor variables are zero, suggesting that all
# predictor variables are related to 'stator_tooth'.
# 
# Response variable: 'stator_winding'.
# The p values associated with the t statistic of the predictor variable 'i_q' is 0.299, suggesting that
# that 'stator_winding' is not related to 'i_q'.
# The p values associated with t statistics of all the other the predictor variables are zero, suggesting that
# 'stator_winding' is related to these predictor variables.
# 
# 
# 
# USING TEST DATA:-
# 
# Response variable: 'torque'.
# R2 score: 0.996 
# Excellent fit.
# 'torque' has a strong linear relationship to 'i_q'
# 
# 
# Response variable: 'pm'.
# R2 score: 0.470
# Fair fit.
# 'pm' is influenced by all variables, but less so by 'i_d' and 'i_q'.
# 
# 
# Response variable: 'stator_yoke'.
# R2 score: 0.847
# Very good fit.
# 'stator_yoke' is strongly related to 'coolant' and 'i_d'.
# 
# 
# Response variable: 'stator_tooth'.
# R2 score: 0.703
# Reasonable fit.
# 'stator_tooth' is quite strongly related to 'coolant' and 'i_d'.
# 
# 
# Response variable: 'stator_winding'.
# R2 score: 0.629
# Reasonable fit.
# 'stator_winding' is quite strongly related to 'coolant' and 'i_d'.
# 
# 

# # Variance Inflation Factor.

# In[32]:


# Calculate the variance inflation factor (VIF) for the predictor variables.

# Import libraries.
import pandas as pd
import numpy as np
from patsy import dmatrix
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use the ALL data.
sample_motor_data = motor_data.sample(frac = FRAC)

print('Predictor variables: ')
print(predictor_variables) # Print out the predictor variables.

features = "+".join(predictor_variables)

# Create the design matrix X dataframe.
X = dmatrix(features, sample_motor_data, return_type='dataframe')

# For each X, calculate VIF and save in dataframe.
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["Features"] = X.columns
print(' ')
print(vif.round(3))
print(' ')
print(' ')


# # The VIF values are generally okay!

# # Create a dataframe of best estimators (models), and their results.

# In[13]:


# Need a dataframe to hold the best estimators (models), and their test results
# for each of the response variables.

names = response_variables
ml_algs = ['MultilinearReg','LinearSVReg', 'RandomforestReg', 'ExtraReg', 'AdaBoostReg', 'KnnReg', 'EnsembleReg']

estimators_df = pd.DataFrame(index = names, columns = ml_algs)

#print(estimators_df)


# In[14]:


# Populate the estimators data frame.
# rows indexed by 'response variable names'.
# columns indexed by 'ML algorithm names'.
# Each element holds a list of [best_estimator, cv_metrics].
# cv_metrics is a list comprising of [r2_score, adjusted_r2_score, rmse].

def populate_estimators_df(best_est_dict, cv_metrics_dict, ml_algorithm):
        for y_var in response_variables:
            best_est = best_est_dict[y_var]
            cv_metrics = cv_metrics_dict[y_var]
            estimator_data = [best_est, cv_metrics]
            estimators_df.loc[y_var, ml_algorithm] = estimator_data
    


# # Cross validation and metrics generation.

# In[15]:


# General model fit and metrics generation.
# Cross validation used to produce the metrics.

# 5 models considered.
# Response variables:
# 'torque' Torque induced by current
# 'pm' (Permanent Magnet surface temperature - representing the rotor temperature).
# 'stator_yoke' (Stator yoke temperature measured with a thermal sensor)
# 'stator_tooth' (Stator tooth temperature measured with a thermal sensor)
# 'stator_winding' (Stator winding temperature measured with a thermal sensor)

from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import f
import statsmodels.api as sm # Loaded in order to calculate the starndard error of the estimators (coefficients).
from sklearn.model_selection import KFold # Used for K fold cross validation.
from statistics import mean, variance, stdev 

kf = KFold(n_splits=5, shuffle=True, random_state=1)


def cross_validation_metrics(data, best_est_dict):

    rows, cols  =  data.shape

    N = rows # The number of observations.
    p = len(predictor_variables) # The number of predictor variables.

    #X = motor_data[predictor_variables]
    X = data[predictor_variables]
    #X = X.to_numpy()   

    cv_metrics_all = []
    for y_var in response_variables:

        print('Response variable: ', y_var)
        print('Predictor variables: ')
        print(predictor_variables) # Print out the predictor variables.
        print('Best estimator: ') # Print out the best estimator (model)
        print(best_est_dict[y_var])
        print(' ')
        
        #y = motor_data[y_var]
        y = data[y_var]
        #print(y)
        
        model = best_est_dict[y_var]
        # Fit model using all the data.
        model.fit(X, y)
        # Predict the response using ALL the data.
        y_pred = model.predict(X)

        print('Summary of the model using ALL data: ')
        R2 = r2_score(y, y_pred)
        print('Coefficient of determination (R2 score):              ', R2)    
        adjusted_R2 = 1 - (1-R2)*(N-1)/(N-p-1)
        print('Adjusted coefficient of determination (Adj R2 score): ', adjusted_R2) 
        # The mean squared error
        mse = mean_squared_error(y, y_pred)
        rmse = math.sqrt(mse)
        #print('Mean squared error: %.4f' % mse) 
        print('Root mean squared error (RSE score):                  ', rmse) 

        # Produce a residual plot using ALL the data.    
        resid = y-y_pred
        colors = 'b'
        area = 1
        plt.scatter(x=y, y=resid, s=area, c=colors, alpha=0.002)
        plt.title('Residual Plot')
        plt.xlabel(y_var)
        plt.ylabel('residual')
        plt.show()
        print(' ')


        R2_scores = []
        adjusted_R2_scores = []
        rse_scores = []

        for train_index, test_index in kf.split(X):
            #print("TRAIN:", train_index, "TEST:", test_index) 

            #X_train, X_test = X[train_index], X[test_index]
            #y_train, y_test = y[train_index], y[test_index]
           
            X_train, X_test = X.iloc[list(train_index)], X.iloc[list(test_index)]
            y_train, y_test = y.iloc[list(train_index)], y.iloc[list(test_index)]

            # Fit the model using the training data.
            model.fit(X_train, y_train)

            # Predict the response for the test data.
            y_pred = model.predict(X_test)

            # Calculate metrics on the test data.
            # The coefficient of determination: 1 is perfect prediction. Use test data.
            R2_score = r2_score(y_test, y_pred)
            R2_scores.append(R2_score)
            
            adjusted_R2_score = 1 - (1-R2_score)*(N-1)/(N-p-1)
            adjusted_R2_scores.append(adjusted_R2_score)
            
            # The residual standard error = estimate of the standard deviation of the residuals.
            mse = mean_squared_error(y_test, y_pred)
            rse_score = math.sqrt(mse)
            rse_scores.append(rse_score)

        print('Summary of the model using the TEST data: ')
        mean_R2_score = mean(R2_scores)
        print('R2 score:          ', mean_R2_score)

        mean_adjusted_R2_score = mean(adjusted_R2_scores)        
        print('Adjusted R2 score: ', mean_adjusted_R2_score)         
        
        mean_rse_score = mean(rse_scores)        
        print('RSE score:         ', mean_rse_score)
        print(' ')
        
        
        cv_metrics = [mean_R2_score, mean_adjusted_R2_score, mean_rse_score]
        cv_metrics_all.append(cv_metrics)
        
    cv_metrics_dict = dict(zip(response_variables, cv_metrics_all))    
    return(cv_metrics_dict)


# # Grid search with Cross Validation.

# In[16]:



# Grid search with Cross Validation.
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def det_hyper_params_gridCV(X, y, model, params):
    grid_search = GridSearchCV(estimator = model, param_grid = params, cv=5,scoring='r2', return_train_score=True)
    grid_search.fit(X, y)
    #print('grid_search.best_params_', grid_search.best_params_)
    #print('grid_search.best_score_', grid_search.best_score_)
    #print('grid_search.best_estimator', grid_search.best_estimator_)
    return(grid_search)
    


# # Randomised search with Cross Validation.

# In[17]:


# Randomised search with Cross Validation.
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def det_hyper_params_randCV(X, y, model, params):
    n_iters = 3 # May need to change this value.
    rand_search = RandomizedSearchCV(estimator = model, param_distributions = params,n_iter= n_iters, cv=5,scoring='r2', return_train_score=True)
    rand_search.fit(X, y)
    #print('rand_search.best_params_', rand_search.best_params_)
    #print('rand_search.best_score_', rand_search.best_score_)
    #print('rand_search.best_estimator', rand_search.best_estimator_)
    return(rand_search)


# # Function to go through the list of response variables.

# In[18]:


def multiple_response_vars(data, model, params, search_type):
    best_estimators = []
    for y_var in response_variables:
        #print('Response variable:')
        #print(y_var)
        #print('Predictor variables:')
        #print(predictor_variables)
        y = data[y_var]
        X = data[predictor_variables]
        if search_type == 'GridSearchCV':
            search = det_hyper_params_gridCV(X, y, model, params)
        elif search_type == 'RandomizedSearchCV':
            search = det_hyper_params_randCV(X, y, model, params)
        else :
            print('Unknown search method.')
        
        best_est = search.best_estimator_
        # print(best_est.score(X,y)) Returns R2 score.
        y_pred = best_est.predict(X)
        r2 = r2_score(y,y_pred)
        #print('R2  :', r2)
        mse = mean_squared_error(y,y_pred)
        #print('MSE :', mse)
        #print(' ')
        best_estimators.append(best_est)
    return(best_estimators)


# # Multiple Linear Regression Model.

# In[19]:


# Fine Tuning of hyperparameters.
# Multiple linear regression model.
# Grid search with cross validation.

import time, datetime
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use ALL the data.
sample_motor_data = motor_data.sample(frac = FRAC)

lin_reg = linear_model.LinearRegression()
params = {'fit_intercept' : [True]} # One choice! 

tic = time.perf_counter()
best_estimators = multiple_response_vars(sample_motor_data, lin_reg, params, 'GridSearchCV')
toc = time.perf_counter()
print('Tuning time : ', str(datetime.timedelta(seconds = toc-tic)))
print(' ')

# Use cross validation using the best estimators.
tic = time.perf_counter()
best_est_dict = dict(zip(response_variables, best_estimators))

cv_metrics_dict = cross_validation_metrics(sample_motor_data, best_est_dict)
#print (cv_metrics_dict)

toc = time.perf_counter()
print('Best model CV time : ', str(datetime.timedelta(seconds = toc-tic)))

populate_estimators_df(best_est_dict, cv_metrics_dict, 'MultilinearReg' )
#print(estimators_df)



# In[40]:


MODEL_Subfolder = 'MultilinearReg'
save_all_models_and_results(best_est_dict, cv_metrics_dict, MODEL_Subfolder)


# In[41]:


# Load in the saved estimators and cv metrics.
# Just need to supply the name of the subfolder.

#MODEL_Subfolder = 'MultilinearReg'
#best_estimators, cv_metrics_all = load_all_models_and_results(MODEL_Subfolder)
#print(best_estimators)
#print(cv_metrics_all)


# In[2]:


# Fine Tuning of hyperparameters.
# Multiple linear regression model.
# Random search with cross validation.

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use ALL the data.
sample_motor_data = motor_data.sample(frac = FRAC)

lin_reg = linear_model.LinearRegression()
params = {'fit_intercept' : [True]} # One choice!

best_estimators = multiple_response_vars(sample_motor_data, lin_reg, params, 'RandomizedSearchCV')

# Use cross validation using the best estimators.
best_est_dict = dict(zip(response_variables, best_estimators))

cv_metrics_dict = cross_validation_metrics(sample_motor_data, best_est_dict)
#print (cv_metrics_dict)
populate_estimators_df(best_est_dict, cv_metrics_dict, 'MultilinearReg' )
#print(estimators_df)



# In[1]:


#MODEL_Subfolder = 'MultilinearReg'
#save_all_models_and_results(best_est_dict, cv_metrics_dict, MODEL_Subfolder)


# In[44]:


# Load in the saved estimators and cv metrics.
# Just need to supply the name of the subfolder.

#MODEL_Subfolder = 'MultilinearReg'
#best_estimators, cv_metrics_all = load_all_models_and_results(MODEL_Subfolder)
#print(best_estimators)
#print(cv_metrics_all)


# #  Support Vector Machine Model.

# In[45]:


# Fine Tuning of hyperparameters.
# Support vector machine regression model.
# Grid search with cross validation.

import time, datetime
from sklearn.svm import LinearSVR

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use ALL the data.
sample_motor_data = motor_data.sample(frac = FRAC)

svm_reg = LinearSVR(random_state = 1, dual = False, loss = 'squared_epsilon_insensitive') 

params = { 'C' : [0.01, 0.1, 1.0, 10.0, 100.0]} 

tic = time.perf_counter()
best_estimators = multiple_response_vars(sample_motor_data, svm_reg, params, 'GridSearchCV')
toc = time.perf_counter()
print('Tuning time : ', str(datetime.timedelta(seconds = toc-tic)))
print(' ')

# Use cross validation using the best estimators.
tic = time.perf_counter()
best_est_dict = dict(zip(response_variables, best_estimators))

cv_metrics_dict = cross_validation_metrics(sample_motor_data, best_est_dict)
#print (cv_metrics_dict)

toc = time.perf_counter()
print('Best model CV time : ', str(datetime.timedelta(seconds = toc-tic)))

populate_estimators_df(best_est_dict, cv_metrics_dict, 'LinearSVReg' )
#print(estimators_df)



# In[46]:


MODEL_Subfolder = 'LinearSVReg'
save_all_models_and_results(best_est_dict, cv_metrics_dict, MODEL_Subfolder)


# In[47]:


# Load in the saved estimators and cv metrics.
# Just need to supply the name of the subfolder.

#MODEL_Subfolder = 'LinearSVReg'
#best_estimators, cv_metrics_all = load_all_models_and_results(MODEL_Subfolder)
#print(best_estimators)
#print(cv_metrics_all)


# In[48]:


# Fine Tuning of hyperparameters.
# Support vector machine regression model.
# Random search with cross validation.

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use ALL the data.
sample_motor_data = motor_data.sample(frac = FRAC)

svm_reg = LinearSVR(random_state = 1, dual = False, loss = 'squared_epsilon_insensitive') 
params = {'C' : np.linspace(0.1, 3.0, 291)} 

best_estimators = multiple_response_vars(sample_motor_data, svm_reg, params, 'RandomizedSearchCV')
best_est_dict = dict(zip(response_variables, best_estimators))
cross_validation_metrics(sample_motor_data, best_est_dict)


# # It is clear that the tuning parameter C has very little effect on the results, and so using the default of C = 1.0 would suffice.
# 
# # Also, there is very little difference in the metric scores between the multilinear regression model, and the linear support vector regression model.

# # Random Forest Regressor. Excellent test results.

# In[49]:


import time, datetime
from sklearn.ensemble import RandomForestRegressor

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use ALL the data.
sample_motor_data = motor_data.sample(frac = FRAC)

params = { 'random_state' : [1],'n_estimators' : [5]} 
rand_reg = RandomForestRegressor( n_jobs=-1, bootstrap = True)

tic = time.perf_counter()
best_estimators = multiple_response_vars(sample_motor_data, rand_reg, params, 'GridSearchCV')
toc = time.perf_counter()
print('Tuning time : ', str(datetime.timedelta(seconds = toc-tic)))
print(' ')

# Use cross validation using the best estimators.
tic = time.perf_counter()
best_est_dict = dict(zip(response_variables, best_estimators))

cv_metrics_dict = cross_validation_metrics(sample_motor_data, best_est_dict)
#print (cv_metrics_dict)

toc = time.perf_counter()
print('Best model CV time : ', str(datetime.timedelta(seconds = toc-tic)))

populate_estimators_df(best_est_dict, cv_metrics_dict, 'RandomforestReg' )
#print(estimators_df)


# In[50]:


MODEL_Subfolder = 'RandomforestReg'
save_all_models_and_results(best_est_dict, cv_metrics_dict, MODEL_Subfolder)


# In[51]:


# Load in the saved estimators and cv metrics.
# Just need to supply the name of the subfolder.

#MODEL_Subfolder = 'RandomforestReg'
#best_estimators, cv_metrics_all = load_all_models_and_results(MODEL_Subfolder)
#print(best_estimators)
#print(cv_metrics_all)


# # Extremely Randomised Tree Regressor. (Much faster to train than regular random forest. ) Also excellent test results - almost as good as random forest.

# In[52]:


# Fine Tuning of hyperparameters.
# Random forest regression using extremely randomised trees.
# Grid search with cross validation.

import time, datetime
from sklearn.ensemble import ExtraTreesRegressor

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use ALL the data.
sample_motor_data = motor_data.sample(frac = FRAC)

params = { 'random_state' : [1],'n_estimators' : [5]} 
ex_tree_reg = ExtraTreesRegressor( n_jobs=-1, bootstrap = True)

tic = time.perf_counter()
best_estimators = multiple_response_vars(sample_motor_data, ex_tree_reg, params, 'GridSearchCV')
toc = time.perf_counter()
print('Tuning time : ', str(datetime.timedelta(seconds = toc-tic)))
print(' ')

# Use cross validation using the best estimators.
tic = time.perf_counter()
best_est_dict = dict(zip(response_variables, best_estimators))

cv_metrics_dict = cross_validation_metrics(sample_motor_data, best_est_dict)
#print (cv_metrics_dict)

toc = time.perf_counter()
print('Best model CV time : ', str(datetime.timedelta(seconds = toc-tic)))

populate_estimators_df(best_est_dict, cv_metrics_dict, 'ExtraReg' )
#print(estimators_df)


# In[53]:


MODEL_Subfolder = 'ExtraReg'
save_all_models_and_results(best_est_dict, cv_metrics_dict, MODEL_Subfolder)


# In[54]:


# Load in the saved estimators and cv metrics.
# Just need to supply the name of the subfolder.

#MODEL_Subfolder = 'ExtraReg'
#best_estimators, cv_metrics_all = load_all_models_and_results(MODEL_Subfolder)
#print(best_estimators)
#print(cv_metrics_all)


# # Adaptive Boosting Regressor. (AdaBoost).

# In[55]:


# Fine Tuning of hyperparameters.
# Adaptive Boosting regression using decison trees.
# Grid search with cross validation.

import time, datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use ALL the data.
sample_motor_data = motor_data.sample(frac = FRAC)

params = { 'random_state' : [1], 'n_estimators' : [5]} 
ada_reg = AdaBoostRegressor(DecisionTreeRegressor()) 

tic = time.perf_counter()
best_estimators = multiple_response_vars(sample_motor_data, ada_reg, params, 'GridSearchCV')
toc = time.perf_counter()
print('Tuning time : ', str(datetime.timedelta(seconds = toc-tic)))
print(' ')

# Use cross validation using the best estimators.
tic = time.perf_counter()
best_est_dict = dict(zip(response_variables, best_estimators))

cv_metrics_dict = cross_validation_metrics(sample_motor_data, best_est_dict)
#print (cv_metrics_dict)

toc = time.perf_counter()
print('Best model CV time : ', str(datetime.timedelta(seconds = toc-tic)))

populate_estimators_df(best_est_dict, cv_metrics_dict, 'AdaBoostReg' )
#print(estimators_df)


# In[56]:


MODEL_Subfolder = 'AdaBoostReg'
save_all_models_and_results(best_est_dict, cv_metrics_dict, MODEL_Subfolder)


# In[57]:


# Load in the saved estimators and cv metrics.
# Just need to supply the name of the subfolder.

#MODEL_Subfolder = 'AdaBoostReg'
#best_estimators, cv_metrics_all = load_all_models_and_results(MODEL_Subfolder)
#print(best_estimators)
#print(cv_metrics_all)


# # K Nearest Neighbours regression.

# In[58]:


# Fine Tuning of hyperparameters.
# K Nearest Neighbours regression.
# Grid search with cross validation.

import time, datetime
from sklearn.neighbors import KNeighborsRegressor
#from sklearn.ensemble import AdaBoostRegressor

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use ALL the data.
sample_motor_data = motor_data.sample(frac = FRAC)

params = {'n_neighbors' : np.arange(1,10,1)} 
knn_reg = KNeighborsRegressor(n_jobs = -1) 

tic = time.perf_counter()
best_estimators = multiple_response_vars(sample_motor_data, knn_reg, params, 'GridSearchCV')
toc = time.perf_counter()
print('Tuning time : ', str(datetime.timedelta(seconds = toc-tic)))
print(' ')

# Use cross validation using the best estimators.
tic = time.perf_counter()
best_est_dict = dict(zip(response_variables, best_estimators))
cv_metrics_dict = cross_validation_metrics(sample_motor_data, best_est_dict)
#print (cv_metrics_dict)
toc = time.perf_counter()
print('Best model CV time : ', str(datetime.timedelta(seconds = toc-tic)))


populate_estimators_df(best_est_dict, cv_metrics_dict, 'KnnReg' )
#print(estimators_df)




# In[59]:


MODEL_Subfolder = 'KnnReg'
save_all_models_and_results(best_est_dict, cv_metrics_dict, MODEL_Subfolder)


# In[60]:


# Load in the saved estimators and cv metrics.
# Just need to supply the name of the subfolder.

#MODEL_Subfolder = 'KnnReg'
#best_estimators, cv_metrics_all = load_all_models_and_results(MODEL_Subfolder)
#print(best_estimators)
#print(cv_metrics_all)


# # Nonlinear SVM. (Takes a long time to complete).

# In[61]:


# THIS TAKES A LONG TIME TO EXECUTE.

# Fine Tuning of hyperparameters.
# Nonlinear support vector machine regression model.
# Grid search with cross validation.

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use ALL the data.
#sample_motor_data = motor_data.sample(frac = FRAC)
#import time, datetime
#from sklearn.svm import SVR

#params = {}
#svm_poly_reg = SVR(kernel="poly", degree=2)

#tic = time.perf_counter()
#best_estimators = multiple_response_vars(sample_motor_data, svm_poly_reg, params, 'GridSearchCV')
#toc = time.perf_counter()
#print('Tuning time : ', str(datetime.timedelta(seconds = toc-tic)))
#print(' ')

# Use cross validation using the best estimators.
#tic = time.perf_counter()
#best_est_dict = dict(zip(response_variables, best_estimators))
#cross_validation_metrics(sample_motor_data, best_est_dict)
#toc = time.perf_counter()
#print('Best model CV time : ', str(datetime.timedelta(seconds = toc-tic)))


# # Ensemble Learning Approach.

# # Combines several regressors to improve predictions.

# In[82]:


# Random forest regression, extra trees regression and adaboost regression are combined.

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import VotingRegressor

import time, datetime

# Use a sample of the data for testing purposes. Set FRAC = 1.0 to use ALL the data.
sample_motor_data = motor_data.sample(frac = FRAC)

reg1 = RandomForestRegressor(n_estimators = 5, bootstrap = True, n_jobs = -1, random_state = 1)
reg2 = ExtraTreesRegressor(n_estimators = 5, bootstrap = True, n_jobs = -1, random_state = 1)
reg3 = AdaBoostRegressor(DecisionTreeRegressor(),n_estimators = 5, random_state = 1)

# Ensemble regressor.
params = {}
ensemble_reg = VotingRegressor([('rfr', reg1), ('xtr', reg2), ('abr', reg3)], weights = [0.2, 0.1, 0.7])

tic = time.perf_counter()
best_estimators = multiple_response_vars(sample_motor_data, ensemble_reg, params, 'GridSearchCV')
toc = time.perf_counter()
print('Tuning time : ', str(datetime.timedelta(seconds = toc-tic)))
print(' ')

# Use cross validation using the best estimators.
tic = time.perf_counter()
best_est_dict = dict(zip(response_variables, best_estimators))
cv_metrics_dict = cross_validation_metrics(sample_motor_data, best_est_dict)
#print (cv_metrics_dict)
toc = time.perf_counter()
print('Best model CV time : ', str(datetime.timedelta(seconds = toc-tic)))


populate_estimators_df(best_est_dict, cv_metrics_dict, 'EnsembleReg' )
#print(estimators_df)


# In[83]:


MODEL_Subfolder = 'EnsembleReg'
save_all_models_and_results(best_est_dict, cv_metrics_dict, MODEL_Subfolder)


# In[64]:


# Load in the saved estimators and cv metrics.
# Just need to supply the name of the subfolder.

#MODEL_Subfolder = 'EnsembleReg'
#best_estimators, cv_metrics_all = load_all_models_and_results(MODEL_Subfolder)
#print(best_estimators)
#print(cv_metrics_all)


# In[ ]:





# # Code to test prediction times for the various machine learning algorithms.

# # Multiple linear regression.

# In[75]:


# Load in the saved estimators and cv metrics.
# Just need to supply the name of the subfolder.

MODEL_Subfolder = 'MultilinearReg'
best_estimators, cv_metrics_all = load_all_models_and_results(MODEL_Subfolder)
#print(best_estimators)
#print(cv_metrics_all)


# In[77]:


import time, datetime

tic = time.perf_counter()
indx=0
for y_var in response_variables:
    tic = time.perf_counter()
    #print(best_estimators[indx])
    model=best_estimators[indx]

    # Note: ONE data point presented.
    sampledata = motor_data[:1]
    #print(sampledata)

    X_test = sampledata.loc[:,predictor_variables]
    #print(X_test)
    y =sampledata.loc[:, y_var]
    #print(y)

    y_pred = model.predict(X_test)
    #print(y_pred)
    indx=indx+1
toc = time.perf_counter()
print('Calculating time : ', str(datetime.timedelta(seconds = toc-tic)))


# # Ensemble regression.

# In[78]:


# Load in the saved estimators and cv metrics.
# Just need to supply the name of the subfolder.

MODEL_Subfolder = 'EnsembleReg'
best_estimators, cv_metrics_all = load_all_models_and_results(MODEL_Subfolder)
#print(best_estimators)
#print(cv_metrics_all)


# In[79]:


import time, datetime

tic = time.perf_counter()
indx=0
for y_var in response_variables:
    tic = time.perf_counter()
    #print(best_estimators[indx])
    model=best_estimators[indx]

    # Note: ONE data point presented.
    sampledata = motor_data[:1]
    #print(sampledata)

    X_test = sampledata.loc[:,predictor_variables]
    #print(X_test)
    y =sampledata.loc[:, y_var]
    #print(y)

    y_pred = model.predict(X_test)
    #print(y_pred)
    indx=indx+1
toc = time.perf_counter()
print('Calculating time : ', str(datetime.timedelta(seconds = toc-tic)))

