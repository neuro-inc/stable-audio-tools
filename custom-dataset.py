import pandas as pd
import os
import logging
import csv
import ast  
import pandas   

ENABLE_RUNTIME_LOGGING = False

# Set up logging specifically for caption processing
caption_logger = logging.getLogger('caption_processor')
caption_logger.setLevel(logging.INFO)

# Create file handler
fh = logging.FileHandler('./dataset/caption_debug.log')
fh.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)

# Add handler to logger
caption_logger.addHandler(fh)

# Load the CSV file with error handling
try:
    # First try with default settings
    _df = pd.read_csv('./dataset/training_metadata.csv')
    caption_logger.info(f"Successfully loaded CSV with {len(_df)} files")
except pandas.errors.ParserError:
    # If that fails, try with more flexible parsing
    caption_logger.info("Initial CSV load failed, trying with flexible parsing...")
    _df = pd.read_csv('./dataset/training_metadata.csv', 
                      quoting=csv.QUOTE_ALL,
                      escapechar='\\',
                      on_bad_lines='warn')
    caption_logger.info(f"Successfully loaded CSV with flexible parsing")

# Create the mapping using filepath and filename
# Note: This might seem like a filename-to-filename mapping, but in practice,
# this pattern allows for flexible caption generation where:
# 1. Each audio file can have multiple semantic representations
# 2. The same file can be described differently in each training iteration
# 3. Captions can be structured as semantic chunks that can be randomly combined
# Now for simplicity, we'll just use the filename as the caption (look training_metadata.csv has filename and filepath)
_caption_map = dict(zip(_df['filepath'], _df['filename']))

def get_custom_metadata(info, audio):
    """Custom metadata function that returns the filename as the prompt"""
    filename = os.path.basename(info.get('path', ''))
    
    if ENABLE_RUNTIME_LOGGING:
        caption_logger.info(f"Looking up caption for filename: {filename}")
    
    caption = _caption_map.get(filename)
    
    if caption:
        if ENABLE_RUNTIME_LOGGING:
            caption_logger.info(f"File: {filename} | Caption: '{caption}'")
        return {'prompt': caption}
    else:
        if ENABLE_RUNTIME_LOGGING:
            caption_logger.info(f"File: {filename} | No caption found")
        return {'prompt': filename}
