# DNA-Promoter-Classifier

A bioinformatics machine learning project for classifying DNA sequences as promoter or non-promoter regions using a convolutional neural network (CNN) implemented in Python with PyTorch.

## Overview

DNA promoters are regions of DNA that initiate gene transcription. Detecting promoter sequences is an important problem in bioinformatics because promoter regions contain characteristic nucleotide patterns and structural relationships that regulate gene expression.

This project uses a convolutional neural network trained on labeled DNA sequences to classify whether a sequence contains promoter patterns.

## Dataset

The dataset consists of:

* 5920 positive promoter sequences
* 5920 negative sequences

Each sequence:

* contains 2000 nucleotides
* uses the symbols:

  * A (adenine)
  * C (cytosine)
  * G (guanine)
  * T (thymine)
  * N (unknown)

Each line in the dataset has the format:

```text
ACGT... 1/0
```

where:

* `1` indicates a promoter sequence
* `0` indicates a non-promoter sequence

## Preprocessing

DNA sequences are converted into numerical form using one-hot encoding:

| Nucleotide | Encoding     |
| ---------- | ------------ |
| A          | [1, 0, 0, 0] |
| C          | [0, 1, 0, 0] |
| G          | [0, 0, 1, 0] |
| T          | [0, 0, 0, 1] |
| N          | [0, 0, 0, 0] |

The encoded sequences are then processed by the neural network.

## Model Architecture

The model consists of:

* multiple 1D convolutional layers
* batch normalization
* ReLU activation
* max pooling
* dropout regularization
* fully connected layers

The CNN learns local nucleotide motifs and higher-level sequence patterns associated with promoter regions.

## Technologies Used

* Python
* PyTorch
* NumPy
* scikit-learn
* matplotlib

## Training

The dataset is split into:

* 70% training
* 15% validation
* 15% testing

Multiple training runs are performed using different random seeds in order to evaluate model stability and performance consistency.

The following hyperparameters are configurable:

* learning rate
* kernel size
* batch size
* network depth
* number of channels

## Results

Example results:

* Mean test accuracy: ~0.78
* Best test accuracy: ~0.81

Training statistics and plots are automatically saved in the `results/` directory.

## Repository Structure

```text
project/
│
├── data.txt
├── source.py
├── results/
│   └── case{i}/
│       ├── test_run{j}.png
│       ├── metadata.txt
│       ├── results.txt
│       └── statistics.txt
├── README.md
└── requirements.txt
```

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
python source.py
```

## Future Improvements

Possible future improvements include:

* reverse-complement data augmentation
* recurrent or transformer-based architectures
* hyperparameter optimization
* larger genomic datasets
* GPU optimization
