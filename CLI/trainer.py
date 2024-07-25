"""File defining the trainer and evaluator of a neural network model"""

import torch
import time

from torch import nn
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from CLI.TUI_menu import metrics_menu

def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, name_model):
    """
    Function that trains a complete model with input data and configurations provided
    :param model: Complete neural model
    :type: MODELSentimentClassifier
    :param data_loader: Function that loads training data
    :type: DataLoader
    :param loss_fn: Error function
    :type: CrossEntropyLoss
    :param optimizer: Optimization function
    :type: AdamW
    :param device: GPU used if available
    :type: Device
    :param scheduler: Function to progressively reduce learning rate
    :type: LambdaLR
    :param name_model: Name of basic technology of the model
    :type: String
    :return: Trained model
    :type: MODELSentimentClassifier
    :return: Upgraded optimizer
    :type: AdamW
    :return: Upgraded function to progressively reduce learning rate
    :type: LambdaLR
    :return: Training accuracy
    :type: Tensor
    :return: Mean error value
    :type: Float64
    """

    #Customization of the console
    console = Console()
    
    #The model is put into training mode
    model = model.train()
    #To store value of labels in each iteration
    tensor_labels = torch.empty(0)
    #To store value of predictions in each iteration
    tensor_predictions = torch.empty(0)
    
    #Start of training time
    start_time_training = time.time()

    #Space in which personalised progress bars exist
    with Progress(SpinnerColumn(spinner_name='bouncingBall'), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("{task.percentage}%")) as progress:
        #The training progress bar is created
        task_training = progress.add_task("Entrenando...", total=len(data_loader))

        for batch in data_loader:
            #Inputs_ids are extracted from batch data and sent to GPU to speed up training
            input_ids = batch['input_ids'].to(device)
            #Attention mask is extracted from batch data and sent to GPU to speed up training
            attention_mask = batch['attention_mask'].to(device)
            #Labels are extracted from batch data and sent to GPU to speed up training
            labels = batch['text_clasification'].to(device).unsqueeze(1)
            #Model outputs are computed
            outputs = model(input_ids = input_ids, attention_mask = attention_mask)

            #DistilBERT model does not return values in same format as other models.
            if name_model != 'DistilBERT':
                #Predictions are calculated
                #If first one is the maximum, change, if second one is the maximum, non-change
                preds = (outputs >= 0.5).long()

                #Labels is added to labels tensor
                tensor_labels = torch.cat((tensor_labels,labels))
                #Predictions made is added to predictions tensor
                tensor_predictions = torch.cat((tensor_predictions,preds))

                #Error calculation
                loss = loss_fn(outputs, labels)
                #Error is back-propagated
                loss.backward()
            else:
                #Predictions are calculated. Maximum of 2 outputs is taken
                #If first one is the maximum, change, if second one is the maximum, non-change
                _, preds = torch.max(outputs, dim = 1)

                #Labels is added to labels tensor
                tensor_labels = torch.cat((tensor_labels,labels))
                #Predictions made is added to predictions tensor
                tensor_predictions = torch.cat((tensor_predictions,preds))

                #Error calculation
                loss = outputs.loss
                #Error is back-propagated
                loss.backward()

            #Gradient is prevented from increasing too much so as not to slow down progress of training with excessively large jumps
            #Gradient value is always kept between -1 and 1.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)

            #Optimizer (weights) is updated.
            optimizer.step()
            #Training rate is updated
            scheduler.step()
            #Gradients are reset for next iteration
            optimizer.zero_grad()

            #The training progress bar is updated
            progress.update(task_training, advance=1)

    #End of training time
    end_time_training = time.time()

    #Total of training time
    total_time_training = end_time_training - start_time_training

    #Calculate the metrics required for the design study
    metrics_model(tensor_labels, tensor_predictions, total_time_training)

    return model, optimizer, scheduler


def eval_model(model, data_loader,device):
    """
    Function that evaluating a complete model with input data and configurations provided
    :param model: Complete neural model
    :type: MODELSentimentClassifier
    :param data_loader: Function that loads training data
    :type: DataLoader
    :param device: GPU used if available
    :type: Device
    :return: Training accuracy
    :type: Tensor
    """

    #Customization of the console
    console = Console()

    #The model is put into evaluating mode
    model = model.eval()
    #To store value of labels in each iteration
    tensor_labels = torch.empty(0)
    #To store value of predictions in each iteration
    tensor_predictions = torch.empty(0)

    #Start of evaluating time
    start_time_evaluating = time.time()

    #Space in which personalised progress bars exist
    with Progress(SpinnerColumn(spinner_name='bouncingBall'), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("{task.percentage}%")) as progress:
        #The evaluating progress bar is created
        task_evaluating = progress.add_task("Evaluando...", total=len(data_loader))

        #It is indicated that no model parameters should be modified
        with torch.no_grad():
            for batch in data_loader:
                #Inputs_ids are extracted from batch data and sent to GPU to speed up evaluation
                input_ids = batch['input_ids'].to(device)
                #Attention mask is extracted from batch data and sent to GPU to speed up evaluation
                attention_mask = batch['attention_mask'].to(device)
                #Labels are extracted from batch data and sent to GPU to speed up evaluation
                labels = batch['text_clasification'].to(device)
                #Model outputs are computed
                outputs = model(input_ids = input_ids, attention_mask = attention_mask)
                #Predictions are calculated (in this case or performed by BERT).Maximum of 2 outputs is taken
                #If first one is the maximum, suicide, if second one is the maximum, non-suicide.
                _, preds = torch.max(outputs, dim = 1)

                #Labels is added to labels tensor
                tensor_labels = torch.cat((tensor_labels,labels))
                #Predictions made is added to predictions tensor
                tensor_predictions = torch.cat((tensor_predictions,preds))

                #The evaluating progress bar is updated
                progress.update(task_evaluating, advance=1)

    #End of evaluating time
    end_time_evaluating = time.time()

    #Total of evaluating time
    total_time_evaluating = end_time_evaluating - start_time_evaluating

    #Calculate the metrics required for the design study
    metrics_model(tensor_labels, tensor_predictions, total_time_evaluating)


def metrics_model(labels, predictions, execution_time):
    """
    Calculates the metrics necessary for testing the performance and evolution of neural model
    :param labels: Original data classification labels
    :type: Tensor
    :param predictions: Data classification labels predicted by the model
    :type: Tensor
    :param execution_time: Execution training or evaluation time
    :type: Float
    :return: Nothing
    """

    #Confusion Matrix
    confusion = confusion_matrix(labels, predictions)

    #Accuracy
    #Return the fraction of correctly classified samples (float)
    accurancy = accuracy_score(labels, predictions)

    #Recall
    #The recall is the ratio tp / (tp + fn)
    #The recall is intuitively the ability of the classifier to find all the positive samples
    recall = recall_score(labels, predictions, average="binary", zero_division = 0)

    #Precision
    #The precision is the ratio tp / (tp + fp)
    #The precision is intuitively the ability of the classifier not to label as positive a sample that is negative
    precision = precision_score(labels, predictions, average="binary", zero_division = 0)

    #F1
    #The F1 score can be interpreted as a harmonic mean of the precision and recall, 
    #where an F1 score reaches its best value at 1 and worst score at 0
    f1 = f1_score(labels, predictions, average="binary", zero_division = 0)

    #Printout of template with results
    metrics_menu(confusion, accurancy, recall, precision, f1, execution_time)