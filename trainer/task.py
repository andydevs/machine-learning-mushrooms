"""
Load data and train model
"""
import tensorflow as tf
import pandas as pd
from .model import MushroomClassifierModel
from argparse import ArgumentParser

# Other constants
DATA_FILE = 'files/data/mushrooms.csv'
MODEL_FILE = 'files/models/saved-model.tf'

# Defaults
DEF_SHUFF_BUFF = 100
DEF_BATCH_SIZE = 200
DEF_REPEAT_NUM = 4
DEF_TRAIN_FRAC = 0.7
DEF_EPOCHS = 10
DEF_DISPLAY_DATA = False


def get_data(
    batch_size=DEF_BATCH_SIZE, 
    shuffle_buffer=DEF_SHUFF_BUFF, 
    repeat_num=DEF_REPEAT_NUM,
    train_frac=DEF_TRAIN_FRAC, 
    display_data=DEF_DISPLAY_DATA):
    """
    Get data from data file
    """
    # Read dataset. Split features and labels
    df = pd.read_csv(DATA_FILE)
    labels = df.pop('class')
    dataset = tf.data.Dataset.from_tensor_slices((dict(df), labels))

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

    # Shuffle, batch, and repeat dataset
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(repeat_num)

    # Display data if display data flag is on
    if display_data:
        for feature_batch, label_batch in dataset.take(1):
            print(f"'class': {label_batch}")
            print('features:')
            for feature, value in feature_batch.items():
                print(f'    {feature:30s}: {value}')

    # Split into training and testing 
    train_num = int(train_frac*len(dataset))
    train_dataset = dataset.take(train_num)
    test_dataset = dataset.skip(train_num)

    # Return dataset
    return train_dataset, test_dataset


def train_and_evaluate_model(train_dataset, test_dataset, epochs=DEF_EPOCHS):
    """
    Train model. Save model afterwards.
    """
    model = MushroomClassifierModel()
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])
    model.fit(train_dataset, epochs=epochs)
    model.evaluate(test_dataset)
    model.summary()
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
        help='Batch size of data')
    parser.add_argument(
        '--shuffle',
        type=int,
        dest='shuffle_buffer',
        default=DEF_SHUFF_BUFF, 
        help='Size of shuffle buffer')
    parser.add_argument(
        '--train-frac',
        type=float,
        dest='train_frac',
        default=DEF_TRAIN_FRAC, 
        help='Fraction of data for training (remainder is testing)')
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
    train_dataset, test_dataset = get_data(
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
        train_frac=args.train_frac,
        display_data=args.display_data)
    train_and_evaluate_model(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        epochs=args.epochs)