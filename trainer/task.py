"""
Load data and train model
"""
import tensorflow as tf
from argparse import ArgumentParser

# Defaults
DEF_BATCH_SIZE = 200


def train_and_evaluate_model(batch_size=DEF_BATCH_SIZE):
    """
    Train and evaluate model. Save model afterwards.
    """
    # Get dataset from csv
    dataset = tf.data.experimental.make_csv_dataset(
        file_pattern='files/data/mushrooms.csv',
        batch_size=batch_size,
        label_name='class')

    # Temporary (check out the dataset)
    for feature_batch, label_batch in dataset.take(1):
        print("'class': {}".format(label_batch))
        print("features:")
        for key, value in feature_batch.items():
            print("  {!r:20s}: {}".format(key, value))


if __name__ == '__main__':
    parser = ArgumentParser(description='Train and evaluate classifier model')
    parser.add_argument(
        '--batch',
        '--batch-size', 
        type=int,
        dest='batch_size',
        default=DEF_BATCH_SIZE, 
        help='Batch size of training data')
    args = parser.parse_args()
    train_and_evaluate_model(
        batch_size=args.batch_size)