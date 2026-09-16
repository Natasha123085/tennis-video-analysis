from flask import Flask
import os

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY']='123456'
    app.config['UPLOAD_FOLDER'] = 'website/static/uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    from .views import views
    from .auth import auth
    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')
    return app