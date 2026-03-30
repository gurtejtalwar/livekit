#!/bin/bash

# Exit immediately if a command fails
set -e

# Change to the dev_containers directory
cd dev_containers || { echo "Directory dev_containers not found"; exit 1; }

# Run docker compose
docker compose up