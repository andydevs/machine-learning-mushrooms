"""
Main flask app
"""
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
from .form import MushroomForm

# Load model
model = tf.keras.models.load_model('files/models/saved-model.tf')

def predict(form_data):
    # Remove submit and csrf token
    del form_data['submit']
    del form_data['csrf_token']
    # Refactor names and replace characters with numpy arrays
    form_data = { 
        k.replace('_', '-'):np.char.array(v)
        for k,v in form_data.items() }

    # Run prediction
    result = model.predict(form_data)

    # Return first instance values
    return { 'e': result[0,0], 'p': result[0,1] }

# Create and configure app
app = Flask(__name__)
app.config.from_object('config')
Bootstrap(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = MushroomForm()
    result = None
    if request.method == 'POST':
        result = predict(form.data)
    return render_template('index.html', form=form, result=result)