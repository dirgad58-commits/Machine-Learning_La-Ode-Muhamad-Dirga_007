#!/bin/bash

# Setup script untuk deployment
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml

# Create models directory
mkdir -p models

echo "Setup complete!"
