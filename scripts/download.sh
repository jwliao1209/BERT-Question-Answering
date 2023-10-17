#!/bin/bash
if [ ! -f data ]; then
    gdown https://drive.google.com/uc?id=1dkO4bizp7S9r7u2WiELmeqOJuNW4NRXh -O data.zip
    unzip data.zip
fi
