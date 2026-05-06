import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

#Loading the original data
df = pd.read_csv('data/raw/german_credit_data.csv')

#Data Cleaning
#Remove redundant index column if it exists
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

#Handle missing values by labeling them 'unknown'
df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')

#Handle Outliers using IQR clipping
numerical_cols = ['Age', 'Credit amount', 'Duration']
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

#Save the cleaned dataset
df.to_csv('data/processed/cleaned_german_credit_data.csv', index=False)

# Split the data (80% Train, 20% Test)
# Stratify ensures the 'Risk' ratio is preserved in both sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Risk'])

# Save the split sets before encoding
train_df.to_csv('data/processed/train_german_credit.csv', index=False)
test_df.to_csv('data/processed/test_german_credit.csv', index=False)

# Feature Scaling and Encoding
# Label encode target: Risk (good = 1, bad = 0)
le = LabelEncoder()
y_train = le.fit_transform(train_df['Risk'])
y_test = le.transform(test_df['Risk'])

# Define column transformations
categorical_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Job']

#ColumnTransformer: Scale numerical and One-Hot Encode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

#Separate features from target
X_train = train_df.drop('Risk', axis=1)
X_test = test_df.drop('Risk', axis=1)

#Fit and transform features
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

#Reconstruct DataFrames with new feature names
cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = list(numerical_cols) + list(cat_names)

final_train = pd.DataFrame(X_train_transformed, columns=all_feature_names)
final_train['Risk'] = y_train

final_test = pd.DataFrame(X_test_transformed, columns=all_feature_names)
final_test['Risk'] = y_test

#Save the final model-ready datasets
final_train.to_csv('data/processed/final_train_scaled_encoded.csv', index=False)
final_test.to_csv('data/processed/final_test_scaled_encoded.csv', index=False)

print("csv created")