"""
Main flask app
"""
from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from .form import MushroomForm

# Create and configure app
app = Flask(__name__)
app.config.from_object('config')
Bootstrap(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = MushroomForm()
    return render_template('index.html', form=form)