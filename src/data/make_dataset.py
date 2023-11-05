"""Making dataset"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Define the threshold for toxicity
TOXICITY_THRESHOLD = 0.9
DATA_PATH = '../../data/'
# Load the dataset
df = pd.read_csv(DATA_PATH + 'raw/filtered.tsv', delimiter='\t')
df.head()

# Normalize/Standardize numerical features
scaler = StandardScaler()
df[['ref_tox', 'trn_tox', 'similarity', 'lenght_diff']] = scaler.fit_transform(
    df[['ref_tox', 'trn_tox', 'similarity', 'lenght_diff']])

# Text preprocessing
# Lowercasing the text
df['reference'] = df['reference'].str.lower()
df['translation'] = df['translation'].str.lower()

# Remove special characters
df['reference'] = df['reference'].str.replace(r'[^\w\s]+', '', regex=True)
df['translation'] = df['translation'].str.replace(r'[^\w\s]+', '', regex=True)

# Split the data into training, validation, and test sets
# 80% for training and 20% for testing
train_data, test_data = train_test_split(df.drop('id', axis=1), test_size=0.2, random_state=42)

# Save the split datasets to separate CSV files
train_data.to_csv(DATA_PATH + 'interim/train.csv', index=False)
test_data.to_csv(DATA_PATH + 'interim/test.csv', index=False)

filtered_df = df[df['ref_tox'] > TOXICITY_THRESHOLD]
filtered_df.to_csv(DATA_PATH + 'interim/most_toxic_data_2.csv', index=False)
