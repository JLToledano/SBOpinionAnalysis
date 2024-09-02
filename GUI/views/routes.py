from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, session
import requests
import torch

from GUI import train_model, save_model, evaluate_model_function

#Blueprint for the main application routes
main_blueprint = Blueprint('main', __name__)

@main_blueprint.route('/')
def index():
    """
    Route for the home page.
    :return: Rendered home page template
    :rtype: str
    """
    return render_template('index.html')

@main_blueprint.route('/configure', methods=['GET', 'POST'])
def configure():
    """
    Route to configure application parameters.
    Handles GET requests to display the current configuration and POST requests to update configuration.
    :return: Redirects to the home page on success or renders configuration template
    :rtype: str
    """
    if request.method == 'POST':
        configuration_main = current_app.config['configuration_main']

        # Update configuration with form values
        configuration_main['MAX_DATA_LEN'] = int(request.form['MAX_DATA_LEN'])
        configuration_main['BATCH_SIZE'] = int(request.form['BATCH_SIZE'])
        configuration_main['EPOCHS'] = int(request.form['EPOCHS'])
        configuration_main['DATALOADER_NUM_WORKERS'] = int(request.form['DATALOADER_NUM_WORKERS'])
        configuration_main['DROP_OUT_BERT'] = float(request.form['DROP_OUT_BERT'])
        configuration_main['TRANSFER_LEARNING'] = 'TRANSFER_LEARNING' in request.form

        flash("Configuración actualizada exitosamente") # Flash a success message
        return redirect(url_for('main.index')) # Redirect to the home page

    return render_template('configuration.html', config=current_app.config['configuration_main'])

@main_blueprint.route('/dataset', methods=['GET', 'POST'])
def dataset():
    """
    Route to configure the dataset used in the application.
    Handles GET requests to display the current dataset name and POST requests to update it.
    :return: Redirects to the dataset configuration page on success or renders dataset template
    :rtype: str
    """
    if request.method == 'POST':
        file_name = request.form['file_name']
        response = requests.post('http://127.0.0.1:5000/api/assign_new_dataset', json={'file_name': file_name})

        if response.status_code == 200:
            flash("Nombre del dataset actualizado exitosamente") # Flash a success message
        else:
            flash("Error al actualizar el nombre del dataset") # Flash an error message

        return redirect(url_for('main.dataset'))

    #Get current dataset name from API
    response = requests.get('http://127.0.0.1:5000/api/get_dataset_name')
    current_file_name = response.json().get('file_name', 'SBO_dataset.csv')
    return render_template('dataset.html', file_name=current_file_name)

@main_blueprint.route('/help')
def help():
    """
    Route to display the help page.
    :return: Rendered help page template
    :rtype: str
    """
    return render_template('help.html')

@main_blueprint.route('/model', methods=['GET', 'POST'])
def model():
    """
    Route to classify text using a pre-trained model.
    Handles GET requests to display model selection page and POST requests to perform classification.
    :return: Redirects to model page with classification result or renders model template
    :rtype: str
    """
    if request.method == 'POST':
        model_file = request.form['model_file']
        technology = request.form['technology']
        text = request.form['text']

        #Send a POST request to API to classify text
        response = requests.post('http://127.0.0.1:5000/api/model/classify', json={'text': text, 'model_file': model_file, 'technology': technology})
        classification = response.json().get('classification')
        flash(f'Text classified as: {classification}') #Flash classification result
        return redirect(url_for('main.model'))

    #Get list of available models from API
    response = requests.get('http://127.0.0.1:5000/api/model/models')
    model_files = response.json()
    return render_template('model.html', model_files=model_files)

@main_blueprint.route('/train_model', methods=['GET', 'POST'])
def train_model_view():
    """
    Route to train a model using current configuration and dataset.
    Handles POST requests to start the training process.
    :return: Rendered training page template or starts training process
    :rtype: str
    """
    if request.method == 'POST':
        technology = request.form['technology']
        configuration_main = current_app.config['configuration_main']

        #Load the dataset from Flask app configuration
        train_dataset = current_app.config['train_dataset']
        test_dataset = current_app.config['test_dataset']

        #Determine training device (GPU or CPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #Call training function
        return train_model(configuration_main, device, technology, train_dataset, test_dataset)
        
    return render_template('train_model.html')

@main_blueprint.route('/display_metrics')
def display_metrics():
    """
    Route to display training metrics stored in session.
    :return: Rendered metrics page template or redirects to home page if no metrics are found
    :rtype: str
    """
    epoch_metrics = session.get('epoch_metrics')

    if epoch_metrics:
        return render_template('metrics.html', epoch_metrics=epoch_metrics)
    else:
        flash('No se encontraron métricas para mostrar.', 'warning') # Flash a warning if no metrics are found
        return redirect(url_for('main.index'))

@main_blueprint.route('/save_model_after_training', methods=['POST'])
def save_model_after_training():
    """
    Route to save a trained model.
    Handles POST requests to save model with specified name.
    :return: Redirects to metrics display page on success or back to training on failure
    :rtype: str
    """
    model_name = request.form['model_name']

    #Retrieve trained model from Flask app configuration
    model = current_app.config.get('trained_model')

    try:
        save_model(model, model_name) #Save model
        flash('Modelo guardado exitosamente como {}.'.format(model_name), 'success') #Flash a success message
    except ValueError as e:
        flash(str(e), 'danger') #Flash an error message if saving fails
        return redirect(url_for('main.train_model_view'))

    #Redirect to display metrics after saving model
    return redirect(url_for('main.display_metrics'))

@main_blueprint.route('/evaluate', methods=['GET', 'POST'])
def evaluate_model():
    """
    Route to evaluate a pre-trained model using the validation dataset.
    Handles POST requests to start the evaluation process.
    :return: Rendered evaluation page template or starts the evaluation process
    :rtype: str
    """
    if request.method == 'POST':
        model_file = request.form['model_file']
        technology = request.form['technology']

        #Load validation dataset from Flask app configuration
        test_dataset = current_app.config['test_dataset']

        #Determine evaluation device (GPU or CPU)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #Call model evaluation function
        return evaluate_model_function(current_app.config['configuration_main'], device, technology, model_file, test_dataset)

    #Get list of available models from API
    response = requests.get('http://127.0.0.1:5000/api/model/models')
    model_files = response.json()

    return render_template('evaluate_model.html', model_files=model_files)