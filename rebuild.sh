#!/bin/bash

# Build and run the container
docker-compose up --build -d

# Remove dangling images (images with no tag, usually old images)
docker image prune -f

# Optional: remove all unused images
# docker image prune -a -f
