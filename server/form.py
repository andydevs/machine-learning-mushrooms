"""
Create flask form
"""
from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField


class MushroomForm(FlaskForm):
    """
    Form for inputting data into machine learning model
    """
    bruises = SelectField('bruises', choices=[
        ('f', 'no'),
        ('t', 'bruises')
    ])
    cap_color = SelectField('cap-color', choices=[
        ('b', 'buff'),
        ('c', 'cinnamon'),
        ('e', 'red'),
        ('g', 'gray'),
        ('n', 'brown'),
        ('p', 'pink'),
        ('r', 'green'),
        ('u', 'purple'),
        ('w', 'white'),
        ('y', 'yellow')
    ])
    cap_shape = SelectField('cap-shape', choices=[
        ('b', 'bell'),
        ('c', 'conical'),
        ('f', 'flat'),
        ('k', 'knobbed'),
        ('s', 'sunken'),
        ('x', 'convex')
    ])
    cap_surface = SelectField('cap-surface', choices=[
        ('f', 'fibrous'),
        ('g', 'grooves'),
        ('s', 'smooth'),
        ('y', 'scaly')
    ])
    gill_attachment = SelectField('gill-attachment', choices=[
        ('a', 'attached'),
        ('d', 'descending'),
        ('f', 'free'),
        ('n', 'notched')
    ])
    gill_color = SelectField('gill-color', choices=[
        ('b', 'buff'),
        ('e', 'red'),
        ('g', 'gray'),
        ('h', 'chocolate'),
        ('k', 'black'),
        ('n', 'brown'),
        ('o', 'orange'),
        ('p', 'pink'),
        ('r', 'green'),
        ('u', 'purple'),
        ('w', 'white'),
        ('y', 'yellow')
    ])
    gill_size = SelectField('gill-size', choices=[
        ('b', 'broad'),
        ('n', 'narrow')
    ])
    gill_spacing = SelectField('gill-spacing', choices=[
        ('c', 'close'),
        ('d', 'distant'),
        ('w', 'crowded')
    ])
    odor = SelectField('odor', choices=[
        ('a', 'almond'),
        ('c', 'creosote'),
        ('f', 'foul'),
        ('l', 'anise'),
        ('m', 'musty'),
        ('n', 'none'),
        ('p', 'pungent'),
        ('s', 'spicy'),
        ('y', 'fishy')
    ])
    population = SelectField('population', choices=[
        ('a', 'abundant'),
        ('c', 'clustered'),
        ('n', 'numerous'),
        ('s', 'scattered'),
        ('v', 'several'),
        ('y', 'solitary')
    ])
    ring_number = SelectField('ring-number', choices=[
        ('n', 'none'),
        ('o', 'one'),
        ('t', 'two')
    ])
    ring_type = SelectField('ring-type', choices=[
        ('c', 'cobwebby'),
        ('e', 'evanescent'),
        ('f', 'flaring'),
        ('l', 'large'),
        ('n', 'none'),
        ('p', 'pendant'),
        ('s', 'sheathing'),
        ('z', 'zone')
    ])
    spore_print_color = SelectField('spore-print-color', choices=[
        ('b', 'buff'),
        ('h', 'chocolate'),
        ('k', 'black'),
        ('n', 'brown'),
        ('o', 'orange'),
        ('r', 'green'),
        ('u', 'purple'),
        ('w', 'white'),
        ('y', 'yellow')
    ])
    stalk_color_above_ring = SelectField('stalk-color-above-ring', choices=[
        ('b', 'buff'),
        ('c', 'cinnamon'),
        ('e', 'red'),
        ('g', 'gray'),
        ('n', 'brown'),
        ('o', 'orange'),
        ('p', 'pink'),
        ('w', 'white'),
        ('y', 'yellow')
    ])
    stalk_color_below_ring = SelectField('stalk-color-below-ring', choices=[
        ('b', 'buff'),
        ('c', 'cinnamon'),
        ('e', 'red'),
        ('g', 'gray'),
        ('n', 'brown'),
        ('o', 'orange'),
        ('p', 'pink'),
        ('w', 'white'),
        ('y', 'yellow')
    ])
    stalk_root = SelectField('stalk-root', choices=[
        ('?', 'missing'),
        ('b', 'bulbous'),
        ('c', 'club'),
        ('e', 'equal'),
        ('r', 'rooted'),
        ('u', 'cup'),
        ('z', 'rhizomorphs')
    ])
    stalk_shape = SelectField('stalk-shape', choices=[
        ('e', 'enlarging'),
        ('t', 'tapering')
    ])
    stalk_surface_above_ring = SelectField('stalk-surface-above-ring', choices=[
        ('f', 'fibrous'),
        ('k', 'silky'),
        ('s', 'smooth'),
        ('y', 'scaly')
    ])
    stalk_surface_below_ring = SelectField('stalk-surface-below-ring', choices=[
        ('f', 'fibrous'),
        ('k', 'silky'),
        ('s', 'smooth'),
        ('y', 'scaly')
    ])
    veil_color = SelectField('veil-color', choices=[
        ('n', 'brown'),
        ('o', 'orange'),
        ('w', 'white'),
        ('y', 'yellow')
    ])
    veil_type = SelectField('veil-type', choices=[
        ('p', 'partial'),
        ('u', 'universal')
    ])
    ring_number = SelectField('ring-number', choices=[
        ('n', 'none'),
        ('o', 'one'),
        ('t', 'two')
    ])
    ring_type = SelectField('ring-type', choices=[
        ('c', 'cobwebby'),
        ('e', 'evanescent'),
        ('f', 'flaring'),
        ('l', 'large'),
        ('n', 'none'),
        ('p', 'pendant'),
        ('s', 'sheathing'),
        ('z', 'zone')
    ])
    spore_print_color = SelectField('spore-print-color', choices=[
        ('k', 'black'),
        ('n', 'brown'),
        ('b', 'buff'),
        ('h', 'chocolate'),
        ('r', 'green'),
        ('o', 'orange'),
        ('u', 'purple'),
        ('w', 'white'),
        ('y', 'yellow')
    ])
    population = SelectField('population', choices=[
        ('a', 'abundant'),
        ('c', 'clustered'),
        ('n', 'numerous'),
        ('s', 'scattered'),
        ('v', 'several'),
        ('y', 'solitary ')
    ])
    habitat = SelectField('habitat', choices=[
        ('g', 'grasses'),
        ('l', 'leaves'),
        ('m', 'meadows'),
        ('p', 'paths'),
        ('u', 'urban'),
        ('w', 'waste'),
        ('d', 'woods')
    ])
    submit = SubmitField('Submit')