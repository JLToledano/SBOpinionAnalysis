"""File containing different user selectable functions"""

import os
import pathlib
import torch

from transformers import get_linear_schedule_with_warmup
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich.console import Console
from rich.theme import Theme
from rich.prompt import Confirm

from CLI.trainer import train_model, eval_model
from CLI.TUI_menu import models_menu
from mod_transformers.model_selector import CLI_model_selector

def save_model(model):
    """
    Configuration of a trained model is stored in a file for possible future use
    :param model: Trained model
    :type: MODELSentimentClassifier
    :return: Nothing
    """

    name_with_blank_spaces = True
    
    custom_theme = Theme({"error":"red", "required_parameter":"purple"})
    console = Console(theme = custom_theme)

    #If file name specified by user contains blanks, another name is requested
    while name_with_blank_spaces:
        model_name = Prompt.ask("Escoja un nombre para el modelo: ")

        if " " in model_name:
            name_with_blank_spaces = True
            console.print("[error]No se recomienda[/error] el uso de espacios en un [required_parameter]nombre de fichero[/required_parameter]\n")
        else:
            name_with_blank_spaces = False

    #The file path is set
    model_path = pathlib.Path("models", model_name + ".pt")

    #Torch model is stored in selected path
    torch.save(model,os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), model_path))


def load_model():
    """
    A model pre-trained by the user is loaded
    :return: Model
    :type: MODELSentimentClassifier
    """

    custom_theme = Theme({"success":"green", "error":"red", "option":"yellow", "required_parameter":"purple"})
    console = Console(theme = custom_theme)

    #The directory where models are located is established
    path_models = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    address = pathlib.Path(path_models)

    #If model directory is empty, user is prompted to train one first
    if len(os.listdir(address)) == 0:
        console.print("[error]No existen modelos actualmente.[/error] Por favor [required_parameter]entrene uno[/required_parameter] para poder [option]evaluarlo[/option] o [option]utilizarlo[/option]\n")
        model = None

    #If trained models exist, they are listed for user to choose one of them
    else:
        #Available model files are obtained and sorted
        list_models_files = os.listdir(address)
        list_models_files.sort()
        
        #Select the name of one of the models
        name_file = models_menu(list_models_files)

        #Selected model is loaded
        model = torch.load(os.path.join(path_models, name_file))

    return model


def data_loader(dataset,tokenizer,max_len,batch_size,num_workers):
    """
    Adding the necessary structure to the dataset to adapt it to Pytorch
    :param dataset: Generic dataset with data
    :type: Dataset
    :param tokenizer: Function that transforms input data into special codes (tokens)
    :type: Tokenizer
    :param max_len: Maximum number of words accepted by model as input parameter
    :type: Int
    :param batch_size: Lot size. Number of data to be inserted into neural network at a time
    :type: Int
    :param num_workers: Number of processes running in parallel. Analyzes x data in parallel
    :type: Int
    :return: Custom Pytorch dataset
    :type: DataLoader
    """

    dataset.set_tokenizer(tokenizer)
    dataset.set_max_len(max_len)

    #Pytorch-specific DataLoader is created
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def training_model_scratch(configuration_main, device, train_dataset, test_dataset):
    """
    The pre-training model and the additional layers added are trained
    :param configuration_main: Training configurations
    :type: dict[String:String]
    :param device: Calculation optimizer
    :type: Torch Device
    :param train_dataset: Dataset with data for training
    :type: Dataset
    :param test_dataset: Dataset with data for evaluating
    :type: Dataset
    :return: Nothing
    """

    #Basic configuration of a model is loaded
    model_configuration = CLI_model_selector(configuration_main)
    model = model_configuration['model']
    tokenizer = model_configuration['tokenizer']
    name_model = model_configuration['name_model']

    #Creation of Pytorch dataset for training
    train_data_loader = data_loader(train_dataset,tokenizer,configuration_main['MAX_DATA_LEN'],configuration_main['BATCH_SIZE'],configuration_main['DATALOADER_NUM_WORKERS'])

    #Creation of Pytorch dataset for evaluating
    test_data_loader = data_loader(test_dataset,tokenizer,configuration_main['MAX_DATA_LEN'],configuration_main['BATCH_SIZE'],configuration_main['DATALOADER_NUM_WORKERS'])

    #Model is taken to the GPU if available
    model = model.to(device)

    #Optimizer is created and a learning rate lr is assigned.
    if model_configuration['name_model'] == 'RoBERTa':
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=0.0)
    else:
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0.0)

    #Total number of training iterations
    total_steps = len(train_data_loader) * configuration_main['EPOCHS']

    #Function to reduce the learning rate
    scheduler = get_linear_schedule_with_warmup(
        optimizer, #Optimizer function
        num_warmup_steps = 0, #Number of iterations the model waits for to start reducing the learning rate
        num_training_steps = total_steps #Total number of training steps
    )

    #Error function to be minimized
    loss_fn = nn.BCEWithLogitsLoss()
    
    #For each epoch, the model is trained.
    for epoch in range(configuration_main['EPOCHS']):
        print('Epoch {} de {}'.format(epoch + 1, configuration_main['EPOCHS']))
        print('--------------------')

        #Model training and parameter update
        model, optimizer, scheduler = train_model(
            model, train_data_loader, loss_fn, optimizer, device, scheduler, name_model
        )

        #Model validated
        eval_model(model, test_data_loader, device)
    
    #Trained model is stored
    save_model(model)


def evaluating_model_pretraining(configuration_main, device, test_dataset):
    """
    The pre-training model and the additional layers added are evaluated
    :param configuration_main: Evaluating configurations
    :type: dict[String:String]
    :param device: Calculation optimizer
    :type: Torch Device
    :param test_dataset: Dataset with data for evaluating
    :type: Dataset
    :return: Nothing
    """

    #Pre-trained Torch model is loaded
    model = load_model()

    #If it has been possible to select a model, text is requested and sorted
    if model is not None:
        #Basic configuration of a model is loaded
        model_configuration = CLI_model_selector(configuration_main)
        tokenizer = model_configuration['tokenizer']

        #Creation of Pytorch dataset for evaluating
        test_data_loader = data_loader(test_dataset,tokenizer,configuration_main['MAX_DATA_LEN'],configuration_main['BATCH_SIZE'],configuration_main['DATALOADER_NUM_WORKERS'])

        #Model validated
        eval_model(model, test_data_loader,device)


def use_classify_model(configuration_main, device):
    """
    Use pre-training model with a user message
    :param configuration_main: Evaluating configurations
    :type: dict[String:String]
    :param device: Calculation optimizer
    :type: Torch Device
    :return: Nothing
    """

    console = Console()
    repeat_process = True

    #Pre-trained Torch model is loaded
    model = load_model()

    #If it has been possible to select a model, text is requested and sorted
    if model is not None:
        #Basic configuration of a model is loaded
        model_configuration = CLI_model_selector(configuration_main)
        tokenizer = model_configuration['tokenizer']

        while repeat_process:
            #User mesagge
            text = console.input("\nInserte el texto que quiere clasificar en inglés:\n")

            #Coding of input data
            encoding_text = tokenizer.encode_plus(
                text, #Original message
                max_length = configuration_main['MAX_DATA_LEN'], #Maximum number of tokens (counting special tokens)
                truncation = True, #Ignoring tokens beyond the set number of tokens
                add_special_tokens = True, #Special tokens [CLS], [SEP] and [PAD] added
                return_token_type_ids = False,
                padding = 'max_length', #If total number of tokens is less than the established maximum, it is filled with [PAD] until the maximum is reached
                return_attention_mask = True, #Model is instructed to pay attention only to non-empty tokens during training
                return_tensors = 'pt' #Final result of the encoder in Pytorch numbers
            )

            input_ids = encoding_text['input_ids'].to(device) #Numeric input tokens and special tokens
            attention_mask = encoding_text['attention_mask'].to(device) #Attention mask

            #Model outputs are computed
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)
            #Predictions are calculated. Maximum of 2 outputs is taken
            #If first one is the maximum, Change, if second one is the maximum, Non-Change
            _, preds = torch.max(outputs, dim = 1)

            if preds:
                print("\nClasificación: Change\n")
            else:
                print("\nClasificación: Non-Change\n")

            repeat_process = Confirm.ask("¿Quiere analizar otro mensaje?")


def customize_parameter_configuration(configuration_main):
    """
    Function for changing value of the general training and evaluation settings of neural network
    :param configuration_main: Dictionary with general settings
    :type: dict[String:String]
    :return: General settings modified or same if user decides not to make changes
    :type: dict[String:String]
    """
    
    #User is prompted for maximum data length by displaying the default option
    configuration_main['MAX_DATA_LEN'] = IntPrompt.ask("Por favor indique el número máximo de palabras que admita el modelo", default=configuration_main['MAX_DATA_LEN'])

    #User is prompted for batch size by displaying the default option
    configuration_main['BATCH_SIZE'] = IntPrompt.ask("Por favor indique el tamaño de lote", default=configuration_main['BATCH_SIZE'])

    #User is prompted for number of epochs by displaying the default option
    configuration_main['EPOCHS'] = IntPrompt.ask("Por favor indique el número de épocas (iteraciones)", default=configuration_main['EPOCHS'])

    #User is prompted for number of parallel processes by displaying the default option
    configuration_main['DATALOADER_NUM_WORKERS'] = IntPrompt.ask("Por favor indique el número de procesos paralelos", default=configuration_main['DATALOADER_NUM_WORKERS'])

    #User is prompted for drop out per one by displaying the default option
    configuration_main['DROP_OUT_BERT'] = FloatPrompt.ask("Por favor indique el tanto por uno de drop out para BERT", default=configuration_main['DROP_OUT_BERT'])

    #User is prompted for train only new layers by displaying the default option
    configuration_main['TRANSFER_LEARNING'] = Confirm.ask("¿Deseas entrenear únicamente las nuevas capas añadidas a las tecnologías base?")

    confirm_changes = Confirm.ask("¿Estás seguro de los cambios realizados?")

    #If user confirms changes are perpetuated
    if not confirm_changes:
        #User is given possibility to execute changes again.
        configuration_main = customize_parameter_configuration(configuration_main)
        
    return configuration_main


def assign_new_dataset():
    """
    Function for assigning a new dataset file as the source of the training and evaluation data
    :return: Name of new file
    :type: String
    """

    #User is prompted for a filename by displaying the default option
    file_name = Prompt.ask("Por favor indique el nombre de su fichero", default="SBO_dataset.csv")

    return file_name