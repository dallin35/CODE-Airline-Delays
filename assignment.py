import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
# If any of this libraries is missing from your computer,
# please install them using pip.

filename = 'Flight_Delays_2018.csv'
df = pd.read_csv(filename)

df_filtered = df[df['OP_CARRIER'] == 'DL']
df_filtered = df_filtered.dropna(subset=['ARR_DELAY', 'DEP_DELAY', 'DISTANCE'])
print(df_filtered[['ARR_DELAY', 'DEP_DELAY', 'DISTANCE']].describe())
print(df_filtered[['ARR_DELAY', 'DEP_DELAY', 'DISTANCE']].corr())

# Prepare the data for OLS regression
#dependency variable
Y = df_filtered["ARR_DELAY"]
#independent variables
X = df_filtered[["DEP_DELAY", "DISTANCE"]]
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
print(model.summary())

plt.scatter(df_filtered["DEP_DELAY"], df_filtered["ARR_DELAY"], alpha=0.5)
plt.xlabel("Departure delay (minutes)")
plt.ylabel("Arrival delay (minutes)")
plt.title("Arrival vs Departure Delay")
plt.show()
