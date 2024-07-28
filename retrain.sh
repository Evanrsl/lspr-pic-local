#!/bin/bash

# URL of the retrain endpoint
RETRAIN_URL="http://localhost:5000/retrain"

# Send POST request to retrain the model
curl -X POST $RETRAIN_URL
