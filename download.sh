#!/bin/bash
if [ ! -d data ]; then
    gdown https://drive.google.com/uc?id=1dkO4bizp7S9r7u2WiELmeqOJuNW4NRXh -O data.zip
fi

if [ ! -d best_checkpoints ]; then
    gdown https://drive.google.com/uc?id=1e8A7Lc7R2KRgtDfp80NWoJO9kNM7G9k8 -O best_checkpoints.zip
fi
