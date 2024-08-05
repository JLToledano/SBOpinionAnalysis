from flask import Flask
import __init__ as init
from views.routes import main_blueprint
from api.routes import api_blueprint

app = Flask(__name__)
app.secret_key = 'secretkey'


# Registrando los blueprints
app.register_blueprint(main_blueprint)
app.register_blueprint(api_blueprint, url_prefix='/api')

#Load constants and predefined application parameters
configuration_main = init.load_config_sbopinionanalysis()
app.config['configuration_main'] = configuration_main

if __name__ == '__main__':
    app.run(debug=True)