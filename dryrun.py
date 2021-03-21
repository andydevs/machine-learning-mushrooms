"""
Predict output mushroom given input
"""
import tensorflow as tf
import numpy as np
from random import choice
from argparse import ArgumentParser
from pprint import PrettyPrinter

# Utility (args and pretty printer)
parser = ArgumentParser()
parser.add_argument('--size', dest='size', type=int, default=9)
args = parser.parse_args()
pp = PrettyPrinter()

# Dictionary of possible values
dictionary = {
    'cap-shape': 'bcfks',
    'cap-surface': 'fgys',
    'cap-color': 'nbcgrpuewy',
    'bruises': 'tf',
    'odor': 'alcyfmnps',
    'gill-attachment': 'adfn',
    'gill-spacing': 'cwd',
    'gill-size': 'bn',
    'gill-color': 'knbhgropuewy',
    'stalk-shape': 'et',
    'stalk-root': 'bcuezr?',
    'stalk-surface-above-ring': 'fyks',
    'stalk-surface-below-ring': 'fyks',
    'stalk-color-above-ring': 'nbcgopewy',
    'stalk-color-below-ring': 'nbcgopewy',
    'veil-type': 'pu',
    'veil-color': 'nowy',
    'ring-number': 'not',
    'ring-type': 'ceflnpsz',
    'spore-print-color': 'knbhrouwy',
    'population': 'acnsvy',
    'habitat': 'glmpuwd'
}

# Load model
model = tf.keras.models.load_model('files/models/saved-model.tf')

# Dryrun model with random input
datapoint = { 
    k: np.char.array([choice(v) for i in range(args.size)])
    for k,v in dictionary.items() }
pp.pprint(datapoint)
output = model.predict(datapoint)
values = { 'e':output[:,0], 'p':output[:,1] }
pp.pprint(values)