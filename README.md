## CIS3330-CODE-7-Airline-Delays-Analysis

## Instructions

In this CODE assignment, you are going to put into practice descriptive and predictive analytics on a dataset called `Flight_Delays_2018.csv`. This dataset has information about delayed flights in 2018. Your job is to conduct descriptive and predictive analytics, as explained below.

1. Descriptive Analytics: Conduct a descriptive analysis to find shapes, patterns, and trends in the data. 
 * Your descriptive analysis should help you choose **predictors** (i.e., independent variables) for the delays in the arrivals (**"ARR_DELAY"**).
 * Your descriptive analysis should help you **reduce the scope** of your analysis in terms of airports and/or airlines.
 * In a report, include statistics and visualizations that explain why you chose certain predictors and the scope of your analysis.
2. Predictive Analytics: Create a predictive linear regression model using **"statsmodels.api"** and ordinary least squares (OLS).
 * Your predictive analytics should report a model that tries to predict delays in the arrivals (**"ARR_DELAY"**)
 * Your predictive analytics should return a report explaining how predictors explain arrivals delays.
 * In a report, include statistics and visualizations that explain what significant predictors in your analysis are and how these predictors can be used to create inferences about future observations.
 * Your report should also include 1-2 paragraphs explaining the findings from your predictive analytical model.

### Deliverables 

No automated testing is enabled in this assignment, but any code used for this assignment needs to be in `assignment.py`. For your deliverable, submit the following two things:

* Code repository with Python code and report in document format (e.g., PDF, Word, etc.)
* Descriptive and predictive analytics report also submitted in Blackboard.

### Possible Report Layout
Introduction
Data Overview
Analysis
Conclusion

## Useful Python Code

* Creating an OLS model with statsmodel.api
  * `model = sm.OLS(Y,X).fit()`
* Describing the statsmodel.api OLS model
  * `model.summary()`
* Obtaining parameters of the linear fit.
  * `model.params`
* Plotting scatterplot with matplotlib
  * `plt.scatter(X,Y)`
  * `plt.show()`

## Data Source and License

The data was extracted from the Kaggle Dataset (https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018). The authors of the Kaggle Dataset disclosed that the data was obtained from the Department of Transportation. The extract from the dataset on this assignment can only be used for educational purposes and was obtained by Dr. Villacis Calderon.
