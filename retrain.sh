#!/bin/bash

# Ensure Docker container is running
docker-compose up -d

# Wait for the container to be fully up and running
sleep 10

# URL of the retrain endpoint
RETRAIN_URL="http://localhost:8080/retrain"

# Send POST request to retrain the model
curl -X POST $RETRAIN_URL
