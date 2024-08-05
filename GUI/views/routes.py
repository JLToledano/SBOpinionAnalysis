from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
import requests

main_blueprint = Blueprint('main', __name__)

@main_blueprint.route('/')
def index():
    return render_template('index.html')

@main_blueprint.route('/configure', methods=['GET', 'POST'])
def configure():
    if request.method == 'POST':
        configuration_main = current_app.config['configuration_main']

        # Actualizar configuración con los valores del formulario
        configuration_main['MAX_DATA_LEN'] = int(request.form['MAX_DATA_LEN'])
        configuration_main['BATCH_SIZE'] = int(request.form['BATCH_SIZE'])
        configuration_main['EPOCHS'] = int(request.form['EPOCHS'])
        configuration_main['DATALOADER_NUM_WORKERS'] = int(request.form['DATALOADER_NUM_WORKERS'])
        configuration_main['DROP_OUT_BERT'] = float(request.form['DROP_OUT_BERT'])
        configuration_main['TRANSFER_LEARNING'] = 'TRANSFER_LEARNING' in request.form

        flash("Configuración actualizada exitosamente")
        return redirect(url_for('main.index'))

    return render_template('configuration.html', config=current_app.config['configuration_main'])

@main_blueprint.route('/dataset', methods=['GET', 'POST'])
def dataset():
    if request.method == 'POST':
        file_name = request.form['file_name']
        response = requests.post('http://127.0.0.1:5000/api/assign_new_dataset', json={'file_name': file_name})

        if response.status_code == 200:
            flash("Nombre del dataset actualizado exitosamente")
        else:
            flash("Error al actualizar el nombre del dataset")

        return redirect(url_for('main.dataset'))

    response = requests.get('http://127.0.0.1:5000/api/get_dataset_name')
    current_file_name = response.json().get('file_name', 'SBO_dataset.csv')
    return render_template('dataset.html', file_name=current_file_name)