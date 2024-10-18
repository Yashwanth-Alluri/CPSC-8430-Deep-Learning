# HW2

## Overview

This project implements a video caption generation model using a sequence-to-sequence approach with an attention mechanism. The model generates captions for video inputs.

## Files Included

- **train_yalluri.py**: This is the main training file. It trains the video caption generation model using the MSVD dataset.

- **test_yalluri.py**: This file is used to test the trained model and generate captions for the video inputs in the test dataset.

- **hw2_seq2seq.sh**: This shell script automates the testing process. It runs the test script and saves the generated captions.

- **i2w_yalluri.pickle**: This file contains the word-to-index mapping for decoding the generated captions.

- **model_yalluri.h5**: This is the saved model file after training. It can be loaded for inference.

## How to Run

1. **Train the Model**: 
   Run the following command to train the model:
   python train_yalluri.py

2.	**Test the Model**:
After training, you can test the model and generate captions by running the provided shell script

    bash hw2_seq2seq.sh