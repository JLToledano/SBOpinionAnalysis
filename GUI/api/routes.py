from flask import Blueprint, jsonify, request, current_app
from GUI import dataset_initialize

api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/assign_new_dataset', methods=['POST'])
def assign_new_dataset():
    """
    API endpoint for assigning a new dataset file as the source of the training and evaluation data.
    :return: Name of the new file
    :rtype: JSON
    """
    data = request.get_json()
    dataset_name = data.get('file_name', 'SBO_dataset.csv')
    current_app.config['configuration_main']['FILE_DATASET_NAME'] = dataset_name

    #Reload the dataset after updating the file
    complete_dataset, train_dataset, test_dataset = dataset_initialize(
        current_app.config['configuration_main']['FILE_DATASET_NAME'], 
        current_app.config['configuration_main']['RANDOM_SEED']
    )
    current_app.config['complete_dataset'] = complete_dataset
    current_app.config['train_dataset'] = train_dataset
    current_app.config['test_dataset'] = test_dataset

    return jsonify({"file_name": dataset_name})

@api_blueprint.route('/get_dataset_name', methods=['GET'])
def get_dataset_name():
    """
    API endpoint for getting the current dataset file name.
    :return: Name of the current file
    :rtype: JSON
    """
    return jsonify({"file_name": current_app.config['configuration_main']['FILE_DATASET_NAME']})