"""Core package for the Phishing URL Detector.

Modules
-------
feature_extraction : turn a raw URL string into numeric features
data_loader        : load and standardise labelled URL datasets
preprocessing      : clean data, extract features, split train/test
model_training     : train, compare and persist classification models
model_evaluation   : reports, charts and model comparison
prediction         : load the saved model and classify new URLs
utils              : shared helpers (logging, JSON, validation)
"""

__version__ = "1.0.0"
