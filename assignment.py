
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Read data from file
filename = 'Flight_Delays_2018.csv'
df = pd.read_csv(filename)

# DESCRIPTIVE ANALYTICS

# Display first rows
print("First 5 rows:")
print(df.head())

# Display descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Average arrival delay by airline
print("\nAverage Arrival Delay by Airline:")
print(df.groupby('OP_CARRIER_NAME')['ARR_DELAY'].mean().sort_values())

# Average arrival delay by airport
print("\nTop 10 Airports by Number of Flights:")
print(df['ORIGIN'].value_counts().head(10))

# Select one airline and one airport
airline = df['OP_CARRIER_NAME'].value_counts().idxmax()
airport = df[df['OP_CARRIER_NAME'] == airline]['ORIGIN'].value_counts().idxmax()

print("\nSelected Airline:", airline)
print("Selected Airport:", airport)

df = df[(df['OP_CARRIER_NAME'] == airline) &
        (df['ORIGIN'] == airport)].copy()

# Choose independent variables
columns = ['DEP_DELAY', 'TAXI_OUT', 'DISTANCE', 'ARR_DELAY']
df = df[columns].dropna()

print("\nSelected Data Statistics:")
print(df.describe())

# Correlation matrix
print("\nCorrelation Matrix:")
print(df.corr())

# Graph 1: Arrival Delay Distribution
plt.hist(df['ARR_DELAY'], bins=50)
plt.title('Arrival Delay Distribution')
plt.xlabel('Arrival Delay (Minutes)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('arrival_delay_distribution.png')
plt.show()

# Graph 2: Departure vs Arrival Delay
plt.scatter(df['DEP_DELAY'], df['ARR_DELAY'], alpha=0.3)
plt.title('Departure Delay vs Arrival Delay')
plt.xlabel('Departure Delay (Minutes)')
plt.ylabel('Arrival Delay (Minutes)')
plt.tight_layout()
plt.savefig('departure_vs_arrival.png')
plt.show()

# PREDICTIVE ANALYTICS

# Choose independent variables
predictors = ['DEP_DELAY', 'TAXI_OUT', 'DISTANCE']

# Set up X and Y for linear regression
X = df[predictors]
Y = df['ARR_DELAY']

# Add constant
X = sm.add_constant(X)

# Run OLS regression
model = sm.OLS(Y, X).fit()

# Print regression results
print("\nOLS Regression Results:")
print(model.summary())

# Predicted arrival delays
df['PREDICTED_DELAY'] = model.predict(X)

# Graph 3: Actual vs Predicted Delays
plt.scatter(df['ARR_DELAY'], df['PREDICTED_DELAY'], alpha=0.3)
plt.title('Actual vs Predicted Arrival Delays')
plt.xlabel('Actual Arrival Delay')
plt.ylabel('Predicted Arrival Delay')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.show()

# Print significant predictors
print("\nSignificant Predictors (p < 0.05):")
print(model.pvalues[model.pvalues < 0.05])

print("\nR-squared:", model.rsquared)
print("Adjusted R-squared:", model.rsquared_adj)

# Save regression results
with open('regression_results.txt', 'w') as file:
    file.write(model.summary().as_text())

print("\nAnalysis Complete!")
