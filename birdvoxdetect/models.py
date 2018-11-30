import os
import warnings

with warnings.catch_warnings():
    # Suppress TF and Keras warnings when importing
    warnings.simplefilter("ignore")
    import keras


def _construct_pcen_network():
    keras.models.load_model(network_path)
