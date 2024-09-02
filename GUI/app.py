from flask import Flask
import __init__ as init
from views.routes import main_blueprint
from api.routes import api_blueprint
from api.model import model_bp

import sys
import os

#Add the current directory to sys.path to ensure local modules can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

#Create Flask application instance
app = Flask(__name__)
app.secret_key = 'secretkey'  #Secret key for session management

#Registering blueprints for modular application structure
app.register_blueprint(model_bp, url_prefix='/api/model')
app.register_blueprint(api_blueprint, url_prefix='/api')
app.register_blueprint(main_blueprint)

#Load main configuration and default parameters of application
configuration_main = init.load_config_sbopinionanalysis()
app.config['configuration_main'] = configuration_main

#Initialize and load datasets
complete_dataset, train_dataset, test_dataset = init.dataset_initialize(
    app.config['configuration_main']['FILE_DATASET_NAME'], 
    app.config['configuration_main']['RANDOM_SEED']
)
app.config['complete_dataset'] = complete_dataset
app.config['train_dataset'] = train_dataset
app.config['test_dataset'] = test_dataset

#Run Flask application
if __name__ == '__main__':
    app.run(debug=False)