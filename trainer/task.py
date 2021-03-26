"""
Load data and train model
"""
import tensorflow as tf
import pandas as pd
from argparse import ArgumentParser
from .data import get_data
from .model import MushroomClassifierModel

# Other constants
DATA_FILE = 'files/data/mushrooms.csv'
MODEL_FILE = 'files/models/saved-model.tf'

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
        default=200, 
        help='Batch size of data')
    parser.add_argument(
        '--shuffle',
        type=int,
        dest='shuffle_buffer',
        default=100, 
        help='Size of shuffle buffer')
    parser.add_argument(
        '--repeat',
        type=int,
        dest='repeat_num',
        default=5,
        help='Number of times that the dataset is repeated')
    parser.add_argument(
        '--train-frac',
        type=float,
        dest='train_frac',
        default=0.7, 
        help='Fraction of data for training (remainder is testing)')
    parser.add_argument(
        '--epochs',
        type=int,
        dest='epochs',
        default=10,
        help='Number of training epochs')
    parser.add_argument(
        '--display-data',
        dest='display_data',
        action='store_true',
        default=False,
        help='Display data sample before training')
    parser.add_argument(
        '--no-train',
        dest='train',
        action='store_false',
        default=True,
        help='Don\'t run training. Dry-run process data.')

    # Parse args and train model
    args = parser.parse_args()
    train_dataset, test_dataset = get_data(
        data_file=DATA_FILE,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
        repeat_num=args.repeat_num,
        train_frac=args.train_frac,
        display_data=args.display_data)
    if args.train:
        train_and_evaluate_model(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            epochs=args.epochs)