"""
Load data and train model
"""
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup, CategoryEncoding
from .model import MushroomClassifierModel
from argparse import ArgumentParser

# Other constants
DATA_FILE = 'files/data/mushrooms.csv'
MODEL_FILE = 'files/models/saved-model.h5'

# Defaults
DEF_BATCH_SIZE = 200
DEF_EPOCHS = 10
TRAIN_FRAC = 0.7

def get_data(batch_size=DEF_BATCH_SIZE):
    """
    Get data from data file
    """
    # Get dataset from csv
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=,
        batch_size=batch_size,
        label_name='class')

    # Preprocess label
    def preprocess_label(features, labels):
        """
        Convert string-categorical labels into
        binary one-hot representation
        """
        true_label = tf.constant('p')
        labels = tf.math.equal(labels, true_label)
        labels = tf.cast(labels, tf.uint8)
        labels = tf.one_hot(labels, depth=2)
        return features, labels
    dataset = dataset.map(preprocess_label)

    # Split into training and testing
    train_num = int(TRAIN_FRAC*len(dataset))
    train_dataset = dataset.take(train_num)
    test_dataset = dataset.skip(train_num)

    # Return dataset
    return train_dataset, test_dataset


def train_and_evaluate_model(batch_size=DEF_BATCH_SIZE, epochs=DEF_EPOCHS):
    """
    Train and evaluate model. Save model afterwards.
    """
    train_dataset, test_dataset = get_data(batch_size)
    model = MushroomClassifierModel()
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    model.fit(train_dataset, epochs=epochs)
    model.evaluate(test_dataset)
    model.save(MODEL_FILE)


if __name__ == '__main__':
    # Define argument parser
    parser = ArgumentParser(
        description='Train and evaluate classifier model')
    parser.add_argument(
        '--batch',
        type=int,
        dest='batch_size',
        default=DEF_BATCH_SIZE, 
        help='Batch size of training data')
    parser.add_argument(
        '--epochs',
        type=int,
        dest='epochs',
        default=DEF_EPOCHS,
        help='Number of training epochs'
    )

    # Parse args and train model
    args = parser.parse_args()
    train_and_evaluate_model(
        batch_size=args.batch_size,
        epochs=args.epochs)