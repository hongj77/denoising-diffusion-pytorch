#!/bin/bash

python -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p ./checkpoints
mkdir -p ./samples
