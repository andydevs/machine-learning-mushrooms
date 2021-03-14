"""
Load data and train model
"""
import tensorflow as tf
from .model import MushroomClassifierModel
from argparse import ArgumentParser

# Other constants
DATA_FILE = 'files/data/mushrooms.csv'
MODEL_FILE = 'files/models/saved-model.h5'

# Defaults
DEF_BATCH_SIZE = 200
DEF_EPOCHS = 10
DEF_DISPLAY_DATA = False


def get_data(batch_size=DEF_BATCH_SIZE, display_data=DEF_DISPLAY_DATA):
    """
    Get data from data file
    """
    # Get dataset from csv
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern=DATA_FILE,
        batch_size=batch_size,
        label_name='class')

    # Preprocess labels
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

    # Display data if display data flag is on
    if display_data:
        for feature_batch, label_batch in dataset.take(1):
            print(f"'class': {label_batch}")
            print('features:')
            for feature, value in feature_batch.items():
                print(f'    {feature:30s}: {value}')

    # Return dataset
    return dataset


def train_and_save_model(batch_size=DEF_BATCH_SIZE, epochs=DEF_EPOCHS, display_data=DEF_DISPLAY_DATA):
    """
    Train model. Save model afterwards.
    """
    dataset = get_data(batch_size, display_data)
    model = MushroomClassifierModel()
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    model.fit(dataset, epochs=epochs)
    model.save(MODEL_FILE)


if __name__ == '__main__':
    # Define argument parser
    parser = ArgumentParser(
        description='Train and save classifier model')
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
    parser.add_argument(
        '--display-data',
        dest='display_data',
        action='store_true',
        default=DEF_DISPLAY_DATA,
        help='Display data sample before training')

    # Parse args and train model
    args = parser.parse_args()
    train_and_save_model(
        batch_size=args.batch_size,
        epochs=args.epochs,
        display_data=args.display_data)