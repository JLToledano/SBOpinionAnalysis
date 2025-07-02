from flask import Flask
import __init__ as init
from views.routes import main_blueprint
from api.routes import api_blueprint
from api.model import model_bp
import sys
import os

# For threading
import threading

# Add current directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Robust check: Detect if running in Google Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Create Flask app
app = Flask(__name__)
app.secret_key = 'secretkey'

# Register blueprints
app.register_blueprint(model_bp, url_prefix='/api/model')
app.register_blueprint(api_blueprint, url_prefix='/api')
app.register_blueprint(main_blueprint)

# Load config and datasets
configuration_main = init.load_config_sbopinionanalysis()
app.config['configuration_main'] = configuration_main
complete_dataset, train_dataset, test_dataset = init.dataset_initialize(
    app.config['configuration_main']['FILE_DATASET_NAME'], 
    app.config['configuration_main']['RANDOM_SEED']
)
app.config['complete_dataset'] = complete_dataset
app.config['train_dataset'] = train_dataset
app.config['test_dataset'] = test_dataset

# Function to run Flask in a separate thread
def run_flask():
    app.run(port=5000)

if __name__ == '__main__':
    if IN_COLAB:
        from pyngrok import ngrok
        token = input("🔐 Enter your ngrok authtoken: ")
        ngrok.set_auth_token(token)
        public_url = ngrok.connect(5000)
        print("🚀 Your app is live at:", public_url)

        # Start Flask in a new thread so it doesn't block ngrok output
        thread = threading.Thread(target=run_flask)
        thread.start()
    else:
        app.run(port=5000)