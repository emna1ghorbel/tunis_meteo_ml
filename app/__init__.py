import os
from flask import Flask
from .routes import bp, load_models


def create_app():
    """Create and configure the Flask application.

    Returns:
        Flask: Configured Flask app with registered blueprint and loaded models.
    """
    # Determine absolute paths for templates and static files relative to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    template_path = os.path.join(project_root, 'templates')
    static_path = os.path.join(project_root, 'static')

    app = Flask(__name__, template_folder=template_path, static_folder=static_path)
    # Register routes blueprint
    app.register_blueprint(bp)
    # Load machine learning models once at startup
    app.config['MODELS'] = load_models()
    return app
