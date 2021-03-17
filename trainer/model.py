"""
Define model using keras
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import StringLookup, CategoryEncoding

class MushroomClassifierModel(keras.Model):
    """
    Classify mushrooms based on input data
    """
    def __init__(self):
        """
        Initialize layers
        """
        super(MushroomClassifierModel, self).__init__()

        # Input spec
        self._input = {
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

        # Data preprocessing layers. 
        # Create index based on string vocabulary.
        # Then one hot encode and concatenate resulting vector
        self._index = { 
            k:StringLookup(vocabulary=list(v), name=f'string-lookup-{k}') 
            for k,v in self._input.items() }
        self._one_hot = { 
            k:CategoryEncoding(max_tokens=v.vocab_size(), name=f'category-encode-{k}') 
            for k,v in self._index.items() }
        self._concatenate = layers.Concatenate(axis=1)

        # One dense layer should do it
        self._dense = layers.Dense(2, activation='softmax')

    def call(self, inputs, training=False):
        """
        Run pass through layers with input.
        """
        indexes = { k:self._index[k](v) for k,v in inputs.items() }
        one_hots = { k:self._one_hot[k](v) for k,v in indexes.items() }
        input_vector = self._concatenate(list(one_hots.values()))

        # Feed through dense
        return self._dense(input_vector)