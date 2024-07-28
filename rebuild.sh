#!/bin/bash

# Stop the running containers
docker-compose down

# Remove old images
docker rmi $(docker images -q ecomdiv-ai-autoassign)

# Build the Docker image
docker-compose build

# Start the containers
docker-compose up -d
