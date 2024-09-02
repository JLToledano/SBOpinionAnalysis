import os
import time
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import yaml
import sys
import pathlib

#Add root directory of project to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import current_app, flash, render_template, render_template_string, session
from sklearn.model_selection import train_test_split
import torch

from transformers import get_linear_schedule_with_warmup
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from mod_dataset.dataset import Dataset
from mod_transformers.model_selector import GUI_model_selector
from GUI.api.model import load_model_and_tokenizer

class TrainModelResponse:
    def __init__(self, model, response):
        self.model = model
        self.response = response

def load_config_sbopinionanalysis() -> dict:
    """
    Function that loads general configuration of the application
    :return: Global constants of application
    :rtype: dict[String:String]
    """
    
    configuration_path = os.path.dirname(__file__)
    with open(os.path.join(configuration_path, 'config.yaml'), 'r', encoding='utf-8') as file:
        info_config = yaml.safe_load(file)

    return info_config

def dataset_initialize(file_dataset_name, random_seed):
    """
    Initialization of default dataset.
    :param file_dataset_name: Name of dataset file to load
    :type: str
    :param random_seed: Seed for random operations to ensure reproducibility
    :type: int
    :return: Complete dataset, training dataset, evaluation dataset in correct format
    :rtype: Dataset
    """
    complete_dataset = Dataset()
    train_dataset = Dataset()
    test_dataset = Dataset()

    #Read all data from source file
    raw_data = complete_dataset.read_file(file_dataset_name)

    #Split data into training and evaluation datasets
    train_raw_data, test_raw_data = train_test_split(raw_data, test_size=0.2, random_state=random_seed)

    #Format dataframes to required format
    complete_dataset.format_dataset(raw_data)
    train_dataset.format_dataset(train_raw_data)
    test_dataset.format_dataset(test_raw_data)

    return complete_dataset, train_dataset, test_dataset

def data_loader(dataset, tokenizer, max_len, batch_size, num_workers):
    """
    Adding necessary structure to dataset to adapt it to Pytorch
    :param dataset: Generic dataset with data
    :type: Dataset
    :param tokenizer: Function that transforms input data into special codes (tokens)
    :type: Tokenizer
    :param max_len: Maximum number of words accepted by the model as input parameter
    :type: int
    :param batch_size: Batch size. Number of data points to be inserted into the neural network at a time
    :type: int
    :param num_workers: Number of processes running in parallel. Analyzes x data points in parallel
    :type: int
    :return: Custom Pytorch dataset
    :rtype: DataLoader
    """

    dataset.set_tokenizer(tokenizer)
    dataset.set_max_len(max_len)

    #Create DataLoader specific to Pytorch
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

def train_model(configuration_main, device, technology, train_dataset, test_dataset):
    """
    Function that trains a complete model with input data and provided configurations.
    :param configuration_main: Training configurations
    :type: dict[String:String]
    :param device: Calculation optimizer (e.g., GPU or CPU)
    :type: torch.device
    :param technology: Selected technology for the model (e.g., BERT, ALBERT, RoBERTa)
    :type: str
    :param train_dataset: Dataset with data for training
    :type: Dataset
    :param test_dataset: Dataset with data for evaluating
    :type: Dataset
    :return: Trained model, optimizer, scheduler
    :rtype: Tuple[nn.Module, AdamW, torch.optim.lr_scheduler.LambdaLR]
    """

    #Load basic configuration of a model
    model_configuration = GUI_model_selector(configuration_main, technology)
    model = model_configuration['model']
    tokenizer = model_configuration['tokenizer']
    name_model = model_configuration['name_model']

    #Create Pytorch datasets for training and evaluation
    train_data_loader = data_loader(
        train_dataset, tokenizer, 
        configuration_main['MAX_DATA_LEN'], 
        configuration_main['BATCH_SIZE'], 
        configuration_main['DATALOADER_NUM_WORKERS']
    )

    test_data_loader = data_loader(
        test_dataset, tokenizer, 
        configuration_main['MAX_DATA_LEN'], 
        configuration_main['BATCH_SIZE'], 
        configuration_main['DATALOADER_NUM_WORKERS']
    )

    #Move model to GPU if available
    model = model.to(device)

    #Create optimizer and assign a learning rate
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=5e-4 if name_model == 'RoBERTa' else 1e-4, 
        weight_decay=0.0
    )

    #Total number of training iterations
    total_steps = len(train_data_loader) * configuration_main['EPOCHS']

    #Function to reduce learning rate
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps 
    )

    #Error function to be minimized
    loss_fn = nn.BCEWithLogitsLoss()

    #Initialize list to store metrics for each epoch
    epoch_metrics = []

    #Training loop
    for epoch in range(configuration_main['EPOCHS']):
        print(f'Epoch {epoch + 1} of {configuration_main["EPOCHS"]}')
        print('--------------------')

        #Start training time
        start_time = time.time()

        #Training and updating model parameters
        model.train()
        tensor_labels = torch.empty(0)
        tensor_predictions = torch.empty(0)

        for batch in train_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['text_clasification'].to(device).unsqueeze(1)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = (outputs >= 0).float()

            tensor_labels = torch.cat((tensor_labels, labels))
            tensor_predictions = torch.cat((tensor_predictions, preds))

            loss = loss_fn(outputs, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        #End training time
        end_time = time.time()
        training_time = end_time - start_time

        #Calculate metrics
        cm = confusion_matrix(tensor_labels, tensor_predictions)
        accuracy = accuracy_score(tensor_labels, tensor_predictions)
        recall = recall_score(tensor_labels, tensor_predictions)
        precision = precision_score(tensor_labels, tensor_predictions)
        f1 = f1_score(tensor_labels, tensor_predictions)

        #Store metrics of this epoch in the list
        epoch_metrics.append({
            'training_time': training_time,
            'cm': cm.tolist(),  #Convert confusion matrix to list
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1': f1
        })
        
    #Store metrics in session
    session['epoch_metrics'] = epoch_metrics

    #Temporarily store model in app.config
    current_app.config['trained_model'] = model

    #Prompt for model name after training and save it immediately
    return render_template_string('''
        <form method="POST" action="{{ url_for('main.save_model_after_training') }}">
            <label for="model_name">Model Name:</label>
            <input type="text" id="model_name" name="model_name" required>
            <button type="submit">Save Model</button>
        </form>
    ''')

def save_model(model, model_name):
    """
    Function to save the trained model with a specified name.
    :param model: Trained model
    :type: nn.Module
    :param model_name: Name to save the model file as
    :type: str
    :return: None
    :raises ValueError: If the model name contains spaces
    """
    if " " in model_name:
        raise ValueError("The use of spaces in a file name is not recommended.")

    model_path = pathlib.Path("models_GUI", model_name + ".pt")
    torch.save(model, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), model_path))

def evaluate_model_function(configuration_main, device, technology, model_file, test_dataset):
    """
    Function to evaluate a trained model using a validation dataset.
    :param configuration_main: Evaluation configurations
    :type: dict[String:String]
    :param device: Calculation optimizer (e.g., GPU or CPU)
    :type: torch.device
    :param technology: Selected technology for the model (e.g., BERT, ALBERT, RoBERTa)
    :type: str
    :param model_file: Filename of the trained model to evaluate
    :type: str
    :param test_dataset: Dataset with data for evaluation
    :type: Dataset
    :return: Rendered template with evaluation metrics
    :rtype: str
    """
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models_GUI")
    model_path = os.path.join(models_dir, model_file)

    model, tokenizer = load_model_and_tokenizer(model_path, technology)
    model = model.to(device)

    test_data_loader = data_loader(
        test_dataset, tokenizer, 
        configuration_main['MAX_DATA_LEN'], 
        configuration_main['BATCH_SIZE'], 
        configuration_main['DATALOADER_NUM_WORKERS']
    )

    #Start evaluation time
    start_time = time.time()

    #Evaluate model on test dataset
    tensor_labels = torch.empty(0)
    tensor_predictions = torch.empty(0)

    model.eval()
    with torch.no_grad():
        for batch in test_data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['text_clasification'].to(device).unsqueeze(1)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = (outputs >= 0).float().view(-1)

            tensor_labels = torch.cat((tensor_labels, labels))
            tensor_predictions = torch.cat((tensor_predictions, preds))

    #End evaluation time
    end_time = time.time()
    evaluation_time = end_time - start_time

    #Calculate evaluation metrics
    cm = confusion_matrix(tensor_labels, tensor_predictions)
    accuracy = accuracy_score(tensor_labels, tensor_predictions)
    recall = recall_score(tensor_labels, tensor_predictions)
    precision = precision_score(tensor_labels, tensor_predictions)
    f1 = f1_score(tensor_labels, tensor_predictions)

    #Render page with evaluation metrics
    return render_template('evaluation_result.html', 
                           cm=cm.tolist(), 
                           accuracy=accuracy, 
                           recall=recall, 
                           precision=precision, 
                           f1=f1,
                           evaluation_time=evaluation_time)