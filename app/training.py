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

DATA_DIR = '/app/data/'


def load_constants():
    constants_file = f'{DATA_DIR}/constants.json'
    with open(constants_file, 'r') as file:
        constants = json.load(file)
    return constants


def parse_json_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    # Try parsing as list of dicts
    try:
        data = json.loads(content)
        if isinstance(data, list):
            print(f"Successfully read file: {file_path}")
            return pd.DataFrame(data)
    except json.JSONDecodeError:
        pass
    # Try parsing as concatenated JSON objects
    try:
        data = []
        for line in content.splitlines():
            if line.strip():
                data.append(json.loads(line.strip()))
        print(f"Successfully read file: {file_path}")
        return pd.DataFrame(data)
    except json.JSONDecodeError as e:
        print(f"Skipping file due to JSON parsing error ({file_path}): {e}")
        return None


all_files = [os.path.join(DATA_DIR, 'daily_data', file) for file in os.listdir(
    os.path.join(DATA_DIR, 'daily_data')) if file.endswith('.json')]

# Read and concatenate JSON files
print("Reading and concatenating JSON files...")
df_list = []
for file in all_files:
    try:
        if os.stat(file).st_size == 0:
            print(f"Skipping empty file: {file}")
            continue
        df = parse_json_file(file)
        if df is not None:
            df_list.append(df)
    except (ValueError, EmptyDataError) as e:
        print(f"Skipping file due to error ({file}): {e}")
        continue

if not df_list:
    raise ValueError("No valid JSON files found to process.")

data = pd.concat(df_list, ignore_index=True)
data = data.drop_duplicates(subset=['reqNumber', 'reqLine'])
print("Data read and concatenated successfully.")

# Extract relevant columns and split materialNumber into three columns
df = data[['plantCode', 'purchaseGroupID', 'materialNumber', 'pic']].copy()
df[['product_id_1', 'product_id_2', 'product_id_3']
   ] = df['materialNumber'].str.extract(r'(\d{3})\.(\d{2})\.(\d{4})')

# Load constants
constants = load_constants()
GROUP_SAP = constants['GROUP_SAP']
REGIONAL_SAP = constants['REGIONAL_SAP']
REGIONAL_HEAD = constants['REGIONAL_HEAD']
VEHICLE_SAP = constants['VEHICLE_SAP']

# Combine all relevant PICs into a single list
all_group = [sapid for sublist in GROUP_SAP.values() for sapid in sublist]
all_regional = [sapid for sublist in REGIONAL_SAP.values()
                for sapid in sublist]
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
    # Use the GROUP_SAP dictionary to get possible classes based on group
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
stage1_df = stage1_df[['purchaseGroupID', 'product_id_1',
                       'product_id_2', 'product_id_3', 'plantCode', 'pic']]
stage1_df['plantCode'] = stage1_df['plantCode'].astype(str)

# Training Stage 1
print("Training Stage 1...")
X = stage1_df.drop('pic', axis=1)
y = stage1_df['pic']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.1, random_state=42)

# Defining the column transformer
column_transformer = ColumnTransformer([
    ("ohe", OneHotEncoder(handle_unknown="ignore"), [
     'purchaseGroupID', 'product_id_1', 'product_id_2', 'product_id_3', 'plantCode']),
])

# Defining the XGBoost classifier with specified hyperparameters
clf = xgb.XGBClassifier(
    eval_metric='mlogloss',
    use_label_encoder=False,
    num_class=len(np.unique(y_train)),
    objective='multi:softprob',
    verbosity=1,
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
stage1_directory = os.path.join(DATA_DIR, 'stage1')
os.makedirs(stage1_directory, exist_ok=True)

# Save model and column transformer
clf.save_model(os.path.join(stage1_directory, 'model.json'))
dump(column_transformer, os.path.join(
    stage1_directory, 'column_transformer.joblib'))
with open(os.path.join(stage1_directory, 'variable.json'), 'w') as file:
    json.dump({'pic': label_encoder.classes_.tolist()}, file)


print(f"Model saved in {stage1_directory}")

# Training Stage 2
print("Training Stage 2...")
stage2_df = df[['purchaseGroupID', 'product_id_1',
                'product_id_2', 'product_id_3', 'plantCode', 'pic']].copy()
stage2_df['plantCode'] = stage2_df['plantCode'].astype(str)

regions_df = [stage2_df[stage2_df['pic'].isin(
    REGIONAL_SAP[region])].copy() for region in REGIONAL_SAP.keys()]

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
stage2_directory = os.path.join(DATA_DIR, 'stage2')
os.makedirs(stage2_directory, exist_ok=True)

for i, df in enumerate(regions_df):
    print(f"\nTraining model {i+1} : {list(REGIONAL_SAP.keys())[i]}")
    print(f"number of data {len(df)}")
    X, y, label_encoder = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    column_transformer = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore"), [
         'purchaseGroupID', 'product_id_1', 'product_id_2', 'product_id_3', 'plantCode']),
    ])

    unique_classes = len(np.unique(y_train))
    objective = 'binary:logistic' if unique_classes == 2 else 'multi:softprob'
    num_class = 1 if unique_classes == 2 else unique_classes

    clf = xgb.XGBClassifier(
        eval_metric='mlogloss',
        use_label_encoder=False,
        objective=objective,
        num_class=num_class,
        verbosity=1,
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
    dump(label_encoder, os.path.join(
        stage2_directory, f'label_encoder_{i+1}.joblib'))


print("All models and label encoders saved successfully.")
