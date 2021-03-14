"""
Define model using keras
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import StringLookup, CategoryEncoding

class MushroomClassifierModel(keras.Model):
    def __init__(self):
        super(MushroomClassifierModel, self).__init__()

        # Define index encoders for each input
        self._indx_cap_shape = StringLookup(vocabulary=list('bcfks'))
        self._indx_cap_surface = StringLookup(vocabulary=list('fgys'))
        self._indx_cap_color = StringLookup(vocabulary=list('nbcgrpuewy'))
        self._indx_bruises = StringLookup(vocabulary=list('tf'))
        self._indx_odor = StringLookup(vocabulary=list('alcyfmnps'))
        self._indx_gill_attachment = StringLookup(vocabulary=list('adfn'))
        self._indx_gill_spacing = StringLookup(vocabulary=list('cwd'))
        self._indx_gill_size = StringLookup(vocabulary=list('bn'))
        self._indx_gill_color = StringLookup(vocabulary=list('knbhgropuewy'))

        # Define one-hot encoders for each input
        self._1hot_cap_shape = CategoryEncoding(max_tokens=self._indx_cap_shape.vocab_size())
        self._1hot_cap_surface = CategoryEncoding(max_tokens=self._indx_cap_surface.vocab_size())
        self._1hot_cap_color = CategoryEncoding(max_tokens=self._indx_cap_color.vocab_size())
        self._1hot_bruises = CategoryEncoding(max_tokens=self._indx_bruises.vocab_size())
        self._1hot_odor = CategoryEncoding(max_tokens=self._indx_odor.vocab_size())
        self._1hot_gill_attachment = CategoryEncoding(max_tokens=self._indx_gill_attachment.vocab_size())
        self._1hot_gill_spacing = CategoryEncoding(max_tokens=self._indx_gill_spacing.vocab_size())
        self._1hot_gill_size = CategoryEncoding(max_tokens=self._indx_gill_size.vocab_size())
        self._1hot_gill_color = CategoryEncoding(max_tokens=self._indx_gill_color.vocab_size())

        # Concatenate inputs
        self._concatenate = layers.Concatenate(axis=1)

        # One dense layer should do it
        self._dense = layers.Dense(2, activation='softmax')

    def call(self, inputs, training=False):
        # Encode string categorize
        cap_shape_indx = self._indx_cap_shape(inputs['cap-shape'])
        cap_surface_indx = self._indx_cap_surface(inputs['cap-surface'])
        cap_color_indx = self._indx_cap_color(inputs['cap-color'])
        bruises_indx = self._indx_bruises(inputs['bruises'])
        odor_indx = self._indx_odor(inputs['odor'])
        gill_attachment_indx = self._indx_gill_attachment(inputs['gill-attachment'])
        gill_spacing_indx = self._indx_gill_spacing(inputs['gill-spacing'])
        gill_size_indx = self._indx_gill_size(inputs['gill-size'])
        gill_color_indx = self._indx_gill_color(inputs['gill-color'])

        # One-hot encode
        cap_shape_1hot = self._1hot_cap_shape(cap_shape_indx)
        cap_surface_1hot = self._1hot_cap_surface(cap_surface_indx)
        cap_color_1hot = self._1hot_cap_color(cap_color_indx)
        bruises_1hot = self._1hot_bruises(bruises_indx)
        odor_1hot = self._1hot_odor(odor_indx)
        gill_attachment_1hot = self._1hot_gill_attachment(gill_attachment_indx)
        gill_spacing_1hot = self._1hot_gill_spacing(gill_spacing_indx)
        gill_size_1hot = self._1hot_gill_size(gill_size_indx)
        gill_color_1hot = self._1hot_gill_color(gill_color_indx)

        # Generate input vector
        input_vector = self._concatenate([
            cap_shape_1hot,
            cap_surface_1hot,
            cap_color_1hot,
            bruises_1hot,
            gill_attachment_1hot,
            gill_spacing_1hot,
            gill_size_1hot,
            gill_color_1hot
        ])

        # Feed through dense
        return self._dense(input_vector)