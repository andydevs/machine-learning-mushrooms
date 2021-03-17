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

# Defaults
DEF_SHUFF_BUFF = 100
DEF_BATCH_SIZE = 200
DEF_REPEAT_NUM = 4
DEF_TRAIN_FRAC = 0.7
DEF_EPOCHS = 10
DEF_DISPLAY_DATA = False


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
        '--repeat',
        type=int,
        dest='repeat_num',
        default=DEF_REPEAT_NUM,
        help='Number of times that the dataset is repeated')
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
        help='Number of training epochs')
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
        repeat_num=args.repeat_num,
        train_frac=args.train_frac,
        display_data=args.display_data)
    train_and_evaluate_model(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        epochs=args.epochs)