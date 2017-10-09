"""HTTP API endpoint
"""
import os

from flask import Flask, render_template


APP = Flask(
    __name__,
    static_url_path='',
    static_folder=os.path.join(os.path.dirname(__file__), '..', 'app', 'dist'),
    template_folder=os.path.join(os.path.dirname(__file__), '..', 'app', 'dist')
)


@APP.route("/")
def home():
    """Render single page
    """
    return render_template('index.html')
