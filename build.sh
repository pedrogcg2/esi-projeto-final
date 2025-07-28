#!/user/bin/env bash

set -e


pip install poetry

cd data 
poetry install
poetry run python pipeline.py
cd ..

cd model
poetry install
poetry run python pipeline.py
cd ..

cd api poetry install

