import os
import pandas as pd
import numpy as np
import json
from pandas.errors import EmptyDataError, ParserError
from joblib import dump
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from google.cloud import storage

# Initialize a Cloud Storage client
storage_client = storage.Client()
bucket_name = 'lspr-pic-assign'  # Replace with your actual bucket name

# Function to download files from Cloud Storage
def download_blob(bucket_name, source_blob_name, destination_file_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f'Blob {source_blob_name} downloaded to {destination_file_name}.')

# Function to upload files to Cloud Storage
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f'File {source_file_name} uploaded to {destination_blob_name}.')

# Load constants from GCS
def load_constants_from_gcs():
    constants_file = '/tmp/constants.json'
    download_blob(bucket_name, 'constants/constants.json', constants_file)
    with open(constants_file, 'r') as file:
        constants = json.load(file)
    return constants

# Directory containing the JSON files in GCS
gcs_directory = 'daily_data/'

# Download all JSON files from the GCS directory
blobs = storage_client.list_blobs(bucket_name, prefix=gcs_directory)
all_files = []
for blob in blobs:
    if blob.name.endswith('.json'):
        local_file = f'/tmp/{os.path.basename(blob.name)}'
        download_blob(bucket_name, blob.name, local_file)
        all_files.append(local_file)

# Read and concatenate JSON files
print("Reading and concatenating JSON files...")
df_list = []
for file in all_files:
    try:
        df = pd.read_json(file, lines=True)
        if df is not None:
            df_list.append(df)
    except EmptyDataError:
        continue

data = pd.concat(df_list, ignore_index=True)
data = data.drop_duplicates(subset=['reqNumber', 'reqLine'])
print("Data read and concatenated successfully.")

# Extract relevant columns and split materialNumber into three columns
df = data[['plantCode', 'purchaseGroupID', 'materialNumber', 'pic']].copy()
df[['product_id_1', 'product_id_2', 'product_id_3']] = df['materialNumber'].str.extract(r'(\d{3})\.(\d{2})\.(\d{4})')

# Load constants
constants = load_constants_from_gcs()
GROUP_SAP = constants['GROUP_SAP']
REGIONAL_SAP = constants['REGIONAL_SAP']
REGIONAL_HEAD = constants['REGIONAL_HEAD']
VEHICLE_SAP = constants['VEHICLE_SAP']

# Combine all relevant PICs into a single list
all_group = [sapid for sublist in GROUP_SAP.values() for sapid in sublist]
all_regional = [sapid for sublist in REGIONAL_SAP.values() for sapid in sublist]
stage1_pic = all_group + REGIONAL_HEAD + VEHICLE_SAP

# Filter the DataFrame for relevant PICs and adjust regional PICs
print("Filtering and adjusting PICs...")
stage1_df = df[df['pic'].isin(stage1_pic)].copy()

changed_pic = []
for _, row in stage1_df.iterrows():
    pic = row['pic']
    if pic in all_regional:
        for key, value_list in REGIONAL_SAP.items():
            if pic in value_list:
                row['pic'] = value_list[0]
    changed_pic.append(row)

changed_pic_df = pd.DataFrame(changed_pic).copy()
stage1_df = changed_pic_df[changed_pic_df['pic'].isin(stage1_pic)].copy()

# Keep only PICs with at least 10 occurrences
frequency = stage1_df['pic'].value_counts()
values_to_keep = frequency[frequency >= 10].index
stage1_df = stage1_df[stage1_df['pic'].isin(values_to_keep)].copy()

# Check possible PIC by their group
def stage1_possible_classes(group):
    # Use the PIC_EMAIL dictionary to get possible classes based on group
    possible_classes = GROUP_SAP.get(group, []) + REGIONAL_HEAD + VEHICLE_SAP
    return possible_classes

# Drop rows where PIC is not a possible class
print("Dropping rows with PICs not in possible classes...")
rows_to_drop = []
for idx, row in stage1_df.iterrows():
    possible_classes = stage1_possible_classes(row['purchaseGroupID'])
    if row['pic'] not in possible_classes and row['pic'] in stage1_pic:
        rows_to_drop.append(idx)

stage1_df = stage1_df.drop(rows_to_drop).reset_index(drop=True).copy()

# Clean up columns
stage1_df = stage1_df[['purchaseGroupID', 'product_id_1', 'product_id_2', 'product_id_3', 'plantCode', 'pic']]
stage1_df['plantCode'] = stage1_df['plantCode'].astype(str)

# Training Stage 1
print("Training Stage 1...")
X = stage1_df.drop('pic', axis=1)
y = stage1_df['pic']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Defining the column transformer
column_transformer = ColumnTransformer([
    ("ohe", OneHotEncoder(handle_unknown="ignore"), ['purchaseGroupID', 'product_id_1', 'product_id_2', 'product_id_3', 'plantCode']),
])

# Defining the XGBoost classifier with specified hyperparameters
clf = xgb.XGBClassifier(
    eval_metric='mlogloss',
    use_label_encoder=False,
    num_class=len(np.unique(y_train)),
    objective='multi:softprob',
    verbosity=2,
    learning_rate=0.2,
    max_depth=6,
    subsample=1,
    colsample_bytree=0.8,
    n_estimators=200,
    reg_alpha=0.01,
    reg_lambda=1.0
)

# Training pipeline with XGBoost
model = Pipeline([
    ('transformer', column_transformer),
    ('classifier', clf)
])

model.fit(X_train, y_train)

# Predicting
y_pred = model.predict(X_test)

# Evaluate the accuracy
filtered_accuracy = accuracy_score(y_test, y_pred)
print(f'Filtered Accuracy: {filtered_accuracy}')

# Save the model
print("Saving the model...")
stage1_directory = '/tmp/stage1'
os.makedirs(stage1_directory, exist_ok=True)

# Save model and column transformer
clf.save_model(os.path.join(stage1_directory, 'model.json'))
dump(column_transformer, os.path.join(stage1_directory, 'column_transformer.joblib'))
with open(os.path.join(stage1_directory, 'variable.json'), 'w') as file:
    json.dump({'pic': label_encoder.classes_.tolist()}, file)

# Upload the model to GCS
upload_blob(bucket_name, os.path.join(stage1_directory, 'model.json'), 'stage1/model.json')
upload_blob(bucket_name, os.path.join(stage1_directory, 'column_transformer.joblib'), 'stage1/column_transformer.joblib')
upload_blob(bucket_name, os.path.join(stage1_directory, 'variable.json'), 'stage1/variable.json')

print(f"Model saved in {stage1_directory}")

# Training Stage 2
print("Training Stage 2...")
stage2_df = df[['purchaseGroupID', 'product_id_1', 'product_id_2', 'product_id_3', 'plantCode', 'pic']].copy()
stage2_df['plantCode'] = stage2_df['plantCode'].astype(str)

regions_df = [stage2_df[stage2_df['pic'].isin(REGIONAL_SAP[region])].copy() for region in REGIONAL_SAP.keys()]

for region in regions_df:
    frequency = region['pic'].value_counts()
    values_to_keep = frequency[frequency >= 10].index
    region = region[region['pic'].isin(values_to_keep)].copy()

def preprocess_data(df):
    X = df.drop('pic', axis=1)
    y = df['pic']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded, label_encoder

models = []
output_labels = []
stage2_directory = "/tmp/stage2"
os.makedirs(stage2_directory, exist_ok=True)

for i, df in enumerate(regions_df):
    print(f"\nTraining model {i+1}...")

    X, y, label_encoder = preprocess_data(df)
    output_labels.append(label_encoder.classes_.tolist())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    column_transformer = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore"), ['purchaseGroupID', 'product_id_1', 'product_id_2', 'product_id_3', 'plantCode']),
    ])

    unique_classes = len(np.unique(y_train))
    objective = 'binary:logistic' if unique_classes == 2 else 'multi:softprob'
    num_class = 1 if unique_classes == 2 else unique_classes

    clf = xgb.XGBClassifier(
        eval_metric='mlogloss',
        use_label_encoder=False,
        objective=objective,
        num_class=num_class,
        verbosity=2,
        learning_rate=0.2,
        max_depth=6,
        subsample=1,
        colsample_bytree=0.8,
        n_estimators=200,
        reg_alpha=0.01,
        reg_lambda=1.0
    )
    
    model = Pipeline([
        ('transformer', column_transformer),
        ('classifier', clf)
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for model {i+1}: {accuracy}")
    
    models.append((model, label_encoder))
    dump(model, os.path.join(stage2_directory, f'model_{i+1}.joblib'))
    dump(label_encoder, os.path.join(stage2_directory, f'label_encoder_{i+1}.joblib'))

with open(os.path.join(stage2_directory, 'output_labels.json'), 'w') as f:
    json.dump(output_labels, f)

# Upload the stage 2 models and label encoders to GCS
for i in range(len(models)):
    upload_blob(bucket_name, os.path.join(stage2_directory, f'model_{i+1}.joblib'), f'stage2/model_{i+1}.joblib')
    upload_blob(bucket_name, os.path.join(stage2_directory, f'label_encoder_{i+1}.joblib'), f'stage2/label_encoder_{i+1}.joblib')
upload_blob(bucket_name, os.path.join(stage2_directory, 'output_labels.json'), 'stage2/output_labels.json')

print("All models and label encoders saved successfully.")
