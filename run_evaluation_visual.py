import os
import logging
from pathlib import Path
from openai import OpenAI
from src.translators.LLM import LLM
from src.translators.MTA import MTA
from evaluators.comet import CometEvaluator
import pandas as pd
import json

# Setup the init parameters
logger = logging.getLogger(__name__)

DATASETS = ["hgissbkh/WMT23-Test"] # List of datasets to evaluate on, taken from HuggingFace Datasets
BASEMODELS = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o"] # List of models to evaluate on, taken from OpenAI API "gpt-4", "gpt-4o"

DEFAULT_SYSTEM_PROMPT = """
You are one expert translator and your task is to translate the following text from {source_lang} to {target_lang}:

You should only output the translated text. without any format or markdown text.
Your translated text:
"""

OUTPUT_PATH = Path ("results") # Path to save the results

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

client = OpenAI() # Initialize the OpenAI client

SAVING_PARIOD = 200 # Saving the results every 5 iterations

comet_evaluator = CometEvaluator(logger = logger, gpus=0) # Initialize the Comet evaluator 
#Using the default model for development, update to the latest model for the real evaluation Unbabel/XCOMET-XXL (needs to grant access to the model)

# %%


