<h1 align="center":> Convolutional Neural Networks (CNNs) to Identify Malignant Moles </h1>

<p align="center">
  <img src="/readme_images/credit_delinquency.jpeg" alt="credit_delinquency" width="50%" />
</p>

## Table of Contents 
1. [➤ About The Project](#About-the-Project)
2. [➤ Dataset](#Dataset)
3. [➤ Project Files](#Project-Files)


## About the Project 
This project investigates several different convolutional neural network architectures 
and adapts them to the task of malignant mole identification. 


## Dataset 
The dataset used for this project is publicly-available from 
[Kaggle](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images)


1. <b>SeriousDlin2yrs</b>: Indicates whether a person has experienced 90 days past due delinquency or worse. This is a boolean parameter which is represented as 1 or 0 in the dataset. 1 being that the person has experienced serious delinquency or 0 meaning the person has not experienced serious delinquency.
2. <b>RevolvingUlizationOfUnsecuredLines</b>: Represents the total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits. This value is represented as a percentage and, therefore, ranges from 0 to 1.
3. <b>Age</b>: Age of the borrower in years. This is represented as an integer.
4. <b>NumberOfTime30-59DaysPastDueNotWorse</b>: Number of times that the borrower has been 30-59 days past due but no worse in the last 2 years. This is represented as an integer.
5. <b>DebtRatio</b>: Monthly debt payments, alimony, living costs divided by monthly gross income. This is represented as a percentage and, therefore, ranges from 0 to 1.
6. <b>MonthlyIncome</b>: This is the monthly income of the person. It is represented as a float in units of USD.
7. <b>NumberOfOpenCreditLinesAndLoans</b>: Number of open loans (installment like car lons or mortgages) and lines of credit (e.g. credit cards). This is represented as an integer.
8. <b>NumberOfTimes90DaysLate</b>: Number of times that the borrower has been 90 days or more past due. This is represented as an integer.
9. <b>NumberRealEstateLoansOrLines</b>: Number of mortgage and real estate loans including home equity lines of credit. This is represented as an integer value.
10. <b>NumberOfTime60-89DaysPastDueNotWorse</b>: Number of times borrower has been 60-89 days past due but no worse in the last 2 years. This is represented as an integer value.
11. <b>NumberOfDependents</b>: Number of dependents in family excluding themselves (spouse, children, etc.). This is represented as an integer value.

## Project Files
1. `project_notebook.ipynb` || Jupyter notebook containing all the data wranging, analysis and modeling.
2. `presentation.pdf` || A presentation walking over the entire work of the project and the steps involved.
3. `Part I - Proposal & Business Objectives.pdf` || summary of the project bjectives.
4. `Part II - Data Exploration.pdf` || ummary of the data exploration performed.
5. `Part III - Data Preparation & Modeling.pdf` || summary of the data wrangling and modeling.
6. `Part IV - Model Optimization.pdf` || summary of how the models were optimized.
7. `data/` || includes the data downloaded from Kaggle. 
