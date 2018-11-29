import os
import warnings

with warnings.catch_warnings():
    # Suppress TF and Keras warnings when importing
    warnings.simplefilter("ignore")
    from keras.layers import (
        Add, AveragePooling1D, BatchNormalization, Convolution1D,
        Convolution2D, Dense, Flatten, Input, MaxPooling2D, Permute
    )
    from keras.models import Model
    import keras.regularizers as regularizers
