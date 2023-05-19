# credit-risk-classification

Background
In this Challenge, you’ll use various techniques to train and evaluate a model based on loan risk. You’ll use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

## Instructions
The instructions for this Challenge are divided into the following subsections:

Split the Data into Training and Testing Sets

Create a Logistic Regression Model with the Original Data

Predict a Logistic Regression Model with Resampled Training Data

Write a Credit Risk Analysis Report

## Split the Data into Training and Testing Sets
Open the starter code notebook and use it to complete the following steps:

Read the lending_data.csv data from the Resources folder into a Pandas DataFrame.

Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.

NOTE
A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting.

Split the data into training and testing datasets by using train_test_split.

## Create a Logistic Regression Model with the Original Data
Use your knowledge of logistic regression to complete the following steps:

Fit a logistic regression model by using the training data (X_train and y_train).

Save the predictions for the testing data labels by using the testing feature data (X_test) and the fitted model.

Evaluate the model’s performance by doing the following:

Calculate the accuracy score of the model.

Generate a confusion matrix.

Print the classification report.

Answer the following question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

## Write a Credit Risk Analysis Report
Write a brief report that includes a summary and analysis of the performance of the machine learning models that you used in this homework. You should write this report as the README.md file included in your GitHub repository.

Structure your report by using the report template that Starter_Code.zip includes, ensuring that it contains the following:

An overview of the analysis: Explain the purpose of this analysis.

The results: Using a bulleted list, describe the accuracy score, the precision score, and recall score of the machine learning model.

A summary: Summarize the results from the machine learning model. Include your justification for recommending the model for use by the company. If you don’t recommend the model, justify your reasoning.

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
The purpose of this analysis is to create a supervised machine learning model that accurately predicts if a loan is high-risk or a healthy loan.
* Explain what financial information the data was on, and what you needed to predict.
The data includes the following financial information: loan size, interest rate, borrower income, debt to income, number of accounts, derogatory marks and total debt
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
0    75036
1     2500
Name: loan_status, dtype: int64
* Describe the stages of the machine learning process you went through as part of this analysis.
Split the Data into Training and Testing Sets:
    The data was first separated into labels and features
    The features and labels were split into training/test sets
Created a Logistic Regression Model with the Original Data
Predicted a Logistic Regression Model with Resampled Training Data
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
A classifier was created using the training data
This classifier was then used to make predicitions on the testing data    

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
    - Accuracy: 0.99 (99% accuracy of predictions)
    - Precisioin: for healthy loans = 1.00 for high-risk loans = 0.87
    - Recall score: for healthy loans = 1.00 for high-risk loans = 0.89


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
    - Accuracy: 1.00 (100% accuracy of predictions)
    - Precision: for healthy loans = 1.00 for high-risk loans = 0.87
    - Recall score: for healthy loans = 1.00 for high-risk loans = 1.00

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
Model 2 performs better than Model 1 when it comes to predicting risky loans considering a higher accuracy and recall score. 

I would recommend model 2 since it has a recall score of 1.00 compared to a recall score of 0.89 for model 1.

* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
It is more important to correctly predict actual high-risk loans (1's) which is indicated by the recall score, than to correct for false-positives which is indicated by the precision score.

Since model 2 has a 1.00 recall score, despite the low precision score, I would recommend this model.

If you do not recommend any of the models, please justify your reasoning.