import os
from flask import Flask, request, jsonify
import pandas as pd
from joblib import load, dump
import xgboost as xgb
import numpy as np
import json
from flask_cors import CORS
from datetime import datetime
import subprocess

app = Flask(__name__)
CORS(app)

DATA_DIR = '/app/data/'


def load_constants():
    constants_file = os.path.join(DATA_DIR, 'constants.json')
    with open(constants_file, 'r') as file:
        constants = json.load(file)
    return constants


def save_constants(constants):
    constants_file = os.path.join(DATA_DIR, 'constants.json')
    with open(constants_file, 'w') as file:
        json.dump(constants, file)


def save_predictions(pred):
    today_date = datetime.now().strftime('%Y-%m-%d')
    file_path = os.path.join(DATA_DIR, 'predictions', f'{today_date}.json')
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


def load_stage1_model_local():
    stage1_path = os.path.join(DATA_DIR, 'stage1')
    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join(stage1_path, "model.json"))

    column_transformer = load(os.path.join(
        stage1_path, "column_transformer.joblib"))

    with open(os.path.join(stage1_path, "variable.json"), 'r') as file:
        data = json.load(file)

    pic = data['pic']

    return clf, column_transformer, pic


def load_stage2_model_local(model_index):
    stage2_path = os.path.join(DATA_DIR, 'stage2')
    model = load(os.path.join(stage2_path, f'model_{model_index}.joblib'))
    label_encoder = load(os.path.join(
        stage2_path, f'label_encoder_{model_index}.joblib'))
    return model, label_encoder


def split_product_id(df):
    df[['product_id_1', 'product_id_2', 'product_id_3']
       ] = df['materialNumber'].str.extract(r'(\d{3})\.(\d{2})\.(\d{4})')
    return df


def clean_columns(df):
    return df[['purchaseGroupID', 'product_id_1', 'product_id_2', 'product_id_3', 'plantCode']]


def preprocess_input_data(request_json):
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
    X = df.drop('pic', axis=1, errors='ignore')

    if column_transformer:
        X = column_transformer.transform(X)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    max_probabilities = probabilities.max(axis=1)

    final_pred = []
    for i, (pred, prob) in enumerate(zip(predictions, max_probabilities)):
        translated_label = label_encoder.inverse_transform(
            [pred])[0] if label_encoder else pic[pred]
        final_pred.append(translated_label)

    return final_pred


@app.route("/")
def hello_world():
    return "Hello world!"


@app.route('/update_data', methods=['POST'])
def handle_data():
    today_date = datetime.now().strftime('%Y-%m-%d')
    file_path = os.path.join(DATA_DIR, 'daily_data', f'{today_date}.json')

    new_data_list = request.json

    if not isinstance(new_data_list, list):
        return jsonify({"error": "Data must be a list of dictionaries"}), 400

    for new_data in new_data_list:
        if 'id' not in new_data:
            return jsonify({"error": "Each item must contain an 'id' field"}), 400
        new_data['createdAt'] = datetime.now().isoformat()

    if os.path.exists(file_path):
        with open(file_path, 'r+') as file:
            existing_data = json.load(file)
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
            existing_data.extend(new_data_list)
            file.seek(0)
            file.truncate()
            json.dump(existing_data, file, indent=4)
    else:
        with open(file_path, 'w') as file:
            json.dump(new_data_list, file, indent=4)

    return jsonify({"message": "Data processed successfully"}), 200


@app.route('/stage1', methods=['POST'])
def classify_stage1():
    model, column_transformer, pic = load_stage1_model_local()
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

        save_predictions([json_result])

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
        save_predictions([json_result])
        return output

    regions = ['CKD', 'CRB', 'LMP', 'MDN',
               'MKS', 'PKB', 'PLB', 'PTK', 'SBY', 'SMG']
    if region not in regions:
        return f'Invalid region: {region}', 400

    model_index = regions.index(region) + 1
    model, label_encoder = load_stage2_model_local(model_index)
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

        save_predictions([json_result])

        return output

    except Exception as e:
        return f'Error: {str(e)}', 500

# Endpoints to read and update constants


@app.route('/constants', methods=['GET'])
def get_constants():
    constants = load_constants()
    return jsonify(constants)


@app.route('/constants', methods=['POST'])
def update_constants():
    new_constants = request.json
    constants = load_constants()
    constants.update(new_constants)
    save_constants(constants)
    return jsonify({"message": "Constants updated successfully"}), 200


@app.route('/retrain', methods=['POST'])
def train_model_https():
    try:
        print(f"Received request: {request.json}")
        result = subprocess.run(
            ['python', 'app/training.py'], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            return f"Error: {result.stderr}", 500
        return "Training started successfully", 200
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}", 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9090))
    app.run(host='0.0.0.0', port=port)
