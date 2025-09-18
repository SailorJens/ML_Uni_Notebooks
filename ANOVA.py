import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Simulate data
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'JobTitle': np.random.choice([f'Job_{i}' for i in range(20)], size=n),  # 20 categories
    'Income': np.random.normal(50000, 10000, size=n)
})

# Introduce some actual signal (e.g. higher income for some jobs)
for i in range(20):
    df.loc[df['JobTitle'] == f'Job_{i}', 'Income'] += i * 500  # increase income by job index



# Fit OLS model with categorical predictor
model = ols('Income ~ C(JobTitle)', data=df).fit()

# Run ANOVA
anova_table = sm.stats.anova_lm(model, typ=2)

# Calculate Eta Squared
eta_squared = anova_table['sum_sq']['C(JobTitle)'] / anova_table['sum_sq'].sum()

print(anova_table)
print(f"\nEta Squared: {eta_squared:.4f}")


import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x='JobTitle', y='Income', data=df)
plt.xticks(rotation=45)  # Rotate labels if many categories
plt.title("Income Distribution by Job Title")
plt.show()


sns.violinplot(x='JobTitle', y='Income', data=df)
plt.xticks(rotation=45)
plt.title("Income Distribution by Job Title (Violin Plot)")
plt.show()



mean_income = df.groupby('JobTitle')['Income'].mean().sort_values(ascending=False).reset_index()

sns.barplot(x='JobTitle', y='Income', data=mean_income)
plt.xticks(rotation=45)
plt.title("Mean Income by Job Title")
plt.show()


sns.stripplot(x='JobTitle', y='Income', data=df, jitter=True)
plt.xticks(rotation=45)
plt.title("Income Points by Job Title")
plt.show()


# Plot top 10 job titles with highest mean income
top_jobs = df.groupby('JobTitle')['Income'].mean().sort_values(ascending=False).head(10).index
filtered_df = df[df['JobTitle'].isin(top_jobs)]

sns.boxplot(x='JobTitle', y='Income', data=filtered_df)
plt.xticks(rotation=45)
plt.title("Income Distribution by Top 10 Job Titles")
plt.show()
