import os
from flask import Flask, request, jsonify
import pandas as pd
from joblib import load, dump
import xgboost as xgb
import numpy as np
import json
from flask_cors import CORS
from datetime import datetime
from google.cloud import storage


app = Flask(__name__)
CORS(app)

# Initialize Cloud Storage client
storage_client = storage.Client()
bucket_name = 'lspr-pic-assign'  # Replace with your actual bucket name


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f'File {source_file_name} uploaded to {destination_blob_name}.')


def download_blob(bucket_name, source_blob_name, destination_file_name):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f'Blob {source_blob_name} downloaded to {destination_file_name}.')


def load_constants_from_gcs():
    constants_file = '/tmp/constants.json'
    download_blob(bucket_name, 'constants/constants.json', constants_file)
    with open(constants_file, 'r') as file:
        constants = json.load(file)
    return constants


def save_constants_to_gcs(constants):
    constants_file = '/tmp/constants.json'
    with open(constants_file, 'w') as file:
        json.dump(constants, file)
    upload_blob(bucket_name, constants_file, 'constants/constants.json')


def save_predictions_to_gcs(pred):
    today_date = datetime.now().strftime('%Y-%m-%d')
    file_path = f"/tmp/predict_{today_date}.json"
    timestamp = datetime.now().isoformat()

    wrapped_predictions = [{timestamp: prediction} for prediction in pred]

    if os.path.exists(file_path):
        with open(file_path, 'r+') as file:
            existing_data = json.load(file)
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
            existing_data.extend(wrapped_predictions)
            file.seek(0)
            file.truncate()
            json.dump(existing_data, file, indent=4)
    else:
        with open(file_path, 'w') as file:
            json.dump(wrapped_predictions, file, indent=4)

    upload_blob(bucket_name, file_path, f'predictions/{today_date}.json')


# Load initial constants
constants = load_constants_from_gcs()


def load_stage1_model():
    """
    Load the stage 1 model, column transformer, and PIC labels.
    """
    # Download stage1 model files from GCS
    download_blob(bucket_name, 'stage1/model.json', '/tmp/model.json')
    download_blob(bucket_name, 'stage1/column_transformer.joblib',
                  '/tmp/column_transformer.joblib')
    download_blob(bucket_name, 'stage1/variable.json', '/tmp/variable.json')

    clf = xgb.XGBClassifier()
    clf.load_model("/tmp/model.json")

    column_transformer = load("/tmp/column_transformer.joblib")

    with open("/tmp/variable.json", 'r') as file:
        data = json.load(file)

    pic = data['pic']

    return clf, column_transformer, pic


def load_stage2_model(model_index):
    """
    Load the stage 2 model and label encoder for the given model index.
    """
    # Download stage2 model files from GCS
    download_blob(
        bucket_name, f'stage2/model_{model_index}.joblib', f'/tmp/model_{model_index}.joblib')
    download_blob(
        bucket_name, f'stage2/label_encoder_{model_index}.joblib', f'/tmp/label_encoder_{model_index}.joblib')

    model = load(f'/tmp/model_{model_index}.joblib')
    label_encoder = load(f'/tmp/label_encoder_{model_index}.joblib')
    return model, label_encoder


def split_product_id(df):
    """
    Split product_id into 3 columns.
    """
    df[['product_id_1', 'product_id_2', 'product_id_3']
       ] = df['materialNumber'].str.extract(r'(\d{3})\.(\d{2})\.(\d{4})')
    return df


def clean_columns(df):
    """
    Clean and select necessary columns.
    """
    return df[['purchaseGroupID', 'product_id_1', 'product_id_2', 'product_id_3', 'plantCode']]


def preprocess_input_data(request_json):
    """
    Preprocess the input JSON data and return the DataFrame.
    """
    if not request_json:
        return 'Missing JSON', 400

    try:
        orig_df = pd.DataFrame(request_json)
    except ValueError as e:
        return f'Invalid JSON format: {str(e)}', 400

    df = orig_df.copy()
    df = split_product_id(df)
    df = clean_columns(df)
    return df, orig_df


def transform_and_predict(model, df, label_encoder, column_transformer=None, pic=None):
    """
    Transform the input data, make predictions, and translate them.
    """
    X = df.drop(
        'pic', axis=1, errors='ignore')  # Adjust 'pic' to the actual target column name if different

    if column_transformer:
        X = column_transformer.transform(X)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    max_probabilities = probabilities.max(axis=1)

    final_pred = []
    for i, (pred, prob) in enumerate(zip(predictions, max_probabilities)):
        translated_label = label_encoder.inverse_transform(
            [pred])[0] if label_encoder else pic[pred]
        # print(f"Sample {i}: Predicted label = {translated_label}, Probability = {(prob * 100):.3f}%")
        final_pred.append(translated_label)

    return final_pred


@app.route("/")
def hello_world():
    return "Hello world!"


@app.route('/update_data', methods=['POST'])
def handle_data():
    # Get today's date in the format YYYY-MM-DD
    today_date = datetime.now().strftime('%Y-%m-%d')
    # Define the file path
    file_path = f"/tmp/commit_{today_date}.json"

    # Get the request body
    new_data_list = request.json

    if not isinstance(new_data_list, list):
        return jsonify({"error": "Data must be a list of dictionaries"}), 400

    for new_data in new_data_list:
        if 'id' not in new_data:
            return jsonify({"error": "Each item must contain an 'id' field"}), 400
        # Add the createdAt field with the current timestamp
        new_data['createdAt'] = datetime.now().isoformat()

    if os.path.exists(file_path):
        # If file exists, read the existing data
        with open(file_path, 'r+') as file:
            existing_data = json.load(file)
            if not isinstance(existing_data, list):
                existing_data = [existing_data]

            # Append the new data to the existing data
            existing_data.extend(new_data_list)

            file.seek(0)
            file.truncate()
            json.dump(existing_data, file, indent=4)
    else:
        # If file does not exist, create a new file and save the data
        with open(file_path, 'w') as file:
            json.dump(new_data_list, file, indent=4)

    # Upload the updated data to GCS
    upload_blob(bucket_name, file_path, f'daily_data/{today_date}.json')

    return jsonify({"message": "Data processed successfully"}), 200


@app.route('/stage1', methods=['POST'])
def classify_stage1():
    model, column_transformer, pic = load_stage1_model()
    if model is None or column_transformer is None or pic is None:
        return 'Model not loaded properly', 500

    request_json = request.get_json()
    df, orig_df = preprocess_input_data(request_json)

    if isinstance(df, str):
        return df, orig_df  # df contains error message, orig_df contains error code

    try:
        print(f"predicting {len(request_json)} samples")
        final_pred = transform_and_predict(
            model, df, label_encoder=None, column_transformer=column_transformer, pic=pic)
        orig_df['pic'] = final_pred
        json_result = orig_df.set_index('id')['pic'].to_dict()
        output = json.dumps(json_result, indent=4)

        save_predictions_to_gcs([json_result])

        return output

    except Exception as e:
        return f'Error: {str(e)}', 500


@app.route('/stage2/<region>', methods=['POST'])
def classify_stage2(region):
    request_json = request.get_json()
    if not request_json:
        return 'Missing JSON', 400

    try:
        orig_df = pd.DataFrame(request_json)
    except ValueError as e:
        return f'Invalid JSON format: {str(e)}', 400

    if region == 'CRB':
        orig_df['pic'] = 'DICKY.WN'
        json_result = orig_df.set_index('id')['pic'].to_dict()
        output = json.dumps(json_result, indent=4)
        save_predictions_to_gcs([json_result])
        return output

    regions = ['CKD', 'CRB', 'LMP', 'MKS', 'MDN', 'PKB', 'PLB', 'PTK', 'SMG', 'SBY']
    if region not in regions:
        return f'Invalid region: {region}', 400

    model_index = regions.index(region) + 1
    model, label_encoder = load_stage2_model(model_index)
    if model is None or label_encoder is None:
        return 'Model not loaded properly', 500

    df, orig_df = preprocess_input_data(request_json)

    if isinstance(df, str):
        return df, orig_df  # df contains error message, orig_df contains error code

    try:
        print(f"predicting {len(request_json)} samples")
        final_pred = transform_and_predict(model, df, label_encoder)
        orig_df['pic'] = final_pred
        json_result = orig_df.set_index('id')['pic'].to_dict()
        output = json.dumps(json_result, indent=4)

        save_predictions_to_gcs([json_result])

        return output

    except Exception as e:
        return f'Error: {str(e)}', 500

# Endpoints to read and update constants


@app.route('/constants', methods=['GET'])
def get_constants():
    constants = load_constants_from_gcs()
    return jsonify(constants)


@app.route('/constants', methods=['POST'])
def update_constants():
    new_constants = request.json
    constants = load_constants_from_gcs()
    constants.update(new_constants)
    save_constants_to_gcs(constants)
    return jsonify({"message": "Constants updated successfully"}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
