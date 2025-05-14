#!/bin/bash
DB_HOST=$(yq -r '.database.host' config.yaml)
DB_PORT=$(yq -r '.database.port' config.yaml)
DEBUG=$(yq -r '.debug' config.yaml)

echo "Database Host: $DB_HOST"
echo "Database Port: $DB_PORT"
echo "Debug Mode: $DEBUG"

