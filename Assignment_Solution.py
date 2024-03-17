#!/usr/bin/env python
# coding: utf-8

# In[87]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
sns.set()
from scipy.stats import norm
import scipy.stats as stats


# # Section 1 task 1

# In[63]:


# added the given excel sheet
funnel = pd.read_excel("AssignmentData.xlsx",sheet_name="WorkerFunnel")


# In[64]:


funnel


# In[65]:


# made a copy a the dataframe so that there is no conflict in the original data frame
df = funnel.copy()


# In[66]:


df


# In[67]:


# summary info about the dataframe
df.info()


# In[69]:


# As we can see 30 values are missing from the "Actual productivity" column, so we can delete or fill those with some values.
# We can use the data from the above row or data from the row below or we can simply use the mean of the whole column.
# So here I will be using mean of the whole column so that our overall result will not vary much.
df= df.fillna(value=df['Actual Productivity'].mean())


# In[70]:


df
# So here we have filled all the null cells with the mean values


# In[71]:


df.info()


# # Section 1 task 2

# In[72]:


# Here we have created a new column "Target Achieved" 
# Using apply and lambda function we have specified the function along an axis of a DataFrame and to perform row-wise operations.
# Using if else condition we have provided the output yes or no
df['Target Achieved'] = df.apply(lambda row: 'Yes' if row['Actual Productivity'] > row['Targeted Productivity'] else 'No', axis=1)


# In[73]:


df


# # Section 1 task 2a

# In[74]:


# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Filter dataframe based on the given date range
start_date = pd.Timestamp('2015-01-01')
end_date = pd.Timestamp('2015-03-11')
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# Group by quarter and 'Target Achieved' category and count occurrences
grouped_df = filtered_df.groupby(['Quarter', 'Target Achieved']).size().unstack(fill_value=0)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))

grouped_df.plot(kind='bar', stacked=True, ax=ax)

# Adding labels and title
plt.title('Target Achieved Quarterly')
plt.xlabel('Quarter')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Target Achieved')

plt.tight_layout()
plt.show()


# Interpretation:
# 
# The grouped bar graphs show the distribution of 'Target Achieved' categories ('Yes' and 'No') for each quarter within the given date range.
# By comparing the heights of the bars, you can visually analyze how the level of target achievement varies across different quarters.
# The graph provides insight into the performance of the departments in meeting their productivity targets over quarterly intervals, helping in identifying trends and patterns.

# # Section 1 task 2b

# In[77]:


# Split the data
train = df['Actual Productivity']
test = df['Actual Productivity'].tail(4)

# Train the ARIMA model
arima_model = ARIMA(train, order=(5,1,0)) 
arima_result = arima_model.fit()

# Forecast with ARIMA
arima_forecast = arima_result.forecast(steps=4)

# Calculate Rolling Averages
rolling_avg = train.rolling(window=4).mean().iloc[-1]

# Forecast with Rolling Averages
rolling_avg_forecast = [rolling_avg] * 4

# Plot the results
plt.figure(figsize=(10, 6))

plt.plot(train.index, train, label='Actual Productivity (Training)', color='blue')
plt.plot(test.index, test, label='Actual Productivity (Testing)', color='green')

plt.plot(test.index, arima_forecast, label='ARIMA Forecast', color='red', linestyle='--')
plt.plot(test.index, rolling_avg_forecast, label='Rolling Averages Forecast', color='orange', linestyle='--')

plt.title('Actual Productivity Forecast')
plt.xlabel('Date')
plt.ylabel('Actual Productivity')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# # Section 1 task 2c

# In[79]:


# Calculate MAPE
def calculate_mape(actual, forecast):
    return np.mean(np.abs((actual - forecast) / actual)) * 100

# Calculate MSE
def calculate_mse(actual, forecast):
    return mean_squared_error(actual, forecast)

# Calculate MAPE for ARIMA
arima_mape = calculate_mape(test, arima_forecast)

# Calculate MSE for ARIMA
arima_mse = calculate_mse(test, arima_forecast)

# Calculate MAPE for Rolling Averages
rolling_avg_mape = calculate_mape(test, rolling_avg_forecast)

# Calculate MSE for Rolling Averages
rolling_avg_mse = calculate_mse(test, rolling_avg_forecast)

# Create a summary dataframe
summary_df = pd.DataFrame({
    'Model': ['ARIMA', 'Rolling Averages'],
    'Mean Absolute Percentage Error (MAPE)': [arima_mape, rolling_avg_mape],
    'Mean Squared Error (MSE)': [arima_mse, rolling_avg_mse]
})

print(summary_df)


# # Section 2

# In[80]:


# importing the data
abtest= pd.read_excel("AssignmentData.xlsx",sheet_name="ABTest")


# In[81]:


abtest


# In[82]:


# made a copy a the dataframe so that there is no conflict in the original data frame
df = abtest.copy()


# In[83]:


df.info()


# # Section 2 Task1

# In[84]:


# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'])

# Group by 'Date' and 'Device' and sum the 'Clicks'
clicks_by_device = df.groupby(['Date', 'Device'])['Clicks'].sum().unstack()

# Plotting
clicks_by_device.plot(figsize=(10, 6))
plt.title('Total Number of Clicks by Device')
plt.xlabel('Date')
plt.ylabel('Total Number of Clicks')
plt.legend(title='Device')
plt.grid(True)
plt.show()


# # Section 2 Task 2

# In[86]:


# Given parameters
MDE = 0.03 
alpha = 0.05 
power = 0.80  

# Z-scores for alpha/2 and beta
Z_alpha_2 = norm.ppf(1 - alpha/2)
Z_beta = norm.ppf(power)

# For maximum variability, let's assume p = 0.5
p = 0.5

# Sample size calculation
sample_size = ((Z_alpha_2 * (2*p*(1-p))**0.5 + Z_beta * (p*(1-p) + p*(1-p))) / MDE)**2

# Let's assume a hypothetical value for p1 and  p2
p1 = 0.2  
p2 = 0.25

# Calculate the standard error for each group
se1 = (p1 * (1 - p1) / sample_size)**0.5
se2 = (p2 * (1 - p2) / sample_size)**0.5

# Calculate the Z-statistic for comparing proportions
Z_stat = (p1 - p2) / (se1**2 + se2**2)**0.5

# Calculate the critical Z-value for a two-tailed test
critical_Z = norm.ppf(1 - alpha/2)

# Check if the Z-statistic is greater than the critical Z-value
# If it is, we have sufficient sample size to conclude the test
sufficient_sample_size = abs(Z_stat) > critical_Z

print(f"Required sample size per group: {sample_size:.2f}")
print("We have sufficient sample size to conclude the test." if sufficient_sample_size else "We do not have sufficient sample size to conclude the test.")


# # Section 2 Task 3

# In[89]:


def ab_test(control_visitors, control_conversions, treatment_visitors, treatment_conversions, confidence_level):
    # Check if there are visitors in both groups
    if control_visitors == 0 or treatment_visitors == 0:
        return "Indeterminate (No visitors in one of the groups)"
    
    # Calculate conversion rates for control and treatment groups
    control_conversion_rate = control_conversions / control_visitors
    treatment_conversion_rate = treatment_conversions / treatment_visitors
    
    # Perform hypothesis testing
    z_score = stats.norm.ppf(1 - (1 - confidence_level / 100) / 2)
    pooled_prob = (control_conversions + treatment_conversions) / (control_visitors + treatment_visitors)
    pooled_se = (pooled_prob * (1 - pooled_prob) * (1 / control_visitors + 1 / treatment_visitors)) ** 0.5
    d_hat = treatment_conversion_rate - control_conversion_rate
    margin_of_error = z_score * pooled_se
    
    # Determine if the difference is statistically significant
    if d_hat - margin_of_error > 0:
        return "Experiment Group is Better"
    elif d_hat + margin_of_error < 0:
        return "Control Group is Better"
    else:
        return "Indeterminate"

# Test the function with the given data
control_visitors = 199 + 1413 + 759 + 473 + 183 + 875  
control_conversions = 159 + 142 + 126 + 129 + 289  
treatment_visitors = 0  
treatment_conversions = 0  
confidence_level = 95  

# Perform the A/B test
result = ab_test(control_visitors, control_conversions, treatment_visitors, treatment_conversions, confidence_level)
print("Result of A/B Test:", result)


# In this function:
# 
# -We calculate the conversion rates for the control and treatment groups.
# -We perform hypothesis testing using the Z-test for proportions.
# -We determine if the difference in conversion rates is statistically significant based on the chosen confidence level.
# Since there's no data provided for the treatment group, the function assumes there's no data available for it. As a result, the A/B test can't be performed effectively. To draw conclusions from the A/B test, we need data from both the control and treatment groups. 

# # Section 2 Task 4

# The app I have created in another file named as "asab_test_ap"

# In[ ]:





# In[ ]:




