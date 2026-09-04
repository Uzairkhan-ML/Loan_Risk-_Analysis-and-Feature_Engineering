#Importing Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Loading Dataset
df = pd.read_csv('loan_data.csv') 
print(df.to_string())
print(df.info())

#Handling Missing Values
df['income']=df['income'].fillna(df['income'].median())
print(df.info())
print(df.describe())

#Understanding how variables interact is key to assesment of risk.
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm',fmt='.2f')
plt.title('Correlation Matrix of Loan Features')
plt.show()

#We use a boxplot to see if certain loan amounts are unrealistic compared to the rest of the data 

plt.figure(figsize=(8,4))
sns.boxplot(x=df['loan_amount'])
plt.title('Identifying Outliers in Loan Amounts')
plt.show()

# Clipping the outliers to the 90th Percentile
qu_limit = df['loan_amount'].quantile(0.95)
ql_limit= df['loan_amount'].quantile(0.05)
df['loan_amount_cleaned'] = df['loan_amount'].clip(upper=qu_limit, lower=ql_limit)

plt.figure(figsize=(8,4))
sns.boxplot(x=df['loan_amount_cleaned'])
plt.title('Distribution Outliers in Loan Amounts')
plt.show()  

#Feature Engineering
#We create a Debt-to-Income ratio and a Risk Score to provide more signal for an AI model

# Create new Feature 
df ['dti_ratio']=df['loan_amount_cleaned']/ df['income']

bins=[300,580,670,740,850]
labels=['High Risk','Fair','Good','Execellent']
df['risk_category']=pd.cut(df['credit_score'],bins=bins,labels=labels)

print(df[['income','dti_ratio','risk_category']].head())

#Findings summary Quick analysis of default rates by risk category
summary=df.groupby('risk_category',observed=False)['default'].mean()*100
print(f"Default Percentage by Category:\n{summary}")
