import os
import torch
from flask import Blueprint, current_app, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification, AlbertTokenizer, AlbertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
import sys
sys.path.insert(0, './models')

model_bp = Blueprint('model', __name__)

def load_model_and_tokenizer(model_path, technology):
    """
    Load the model and tokenizer based on the specified technology.
    :param model_path: Path to the pre-trained model file
    :type: str
    :param technology: Technology used to train model (bert, albert, roberta)
    :type: str
    :return: The loaded model and tokenizer
    :rtype: Tuple[PreTrainedModel, PreTrainedTokenizer]
    """
    if technology == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif technology == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    elif technology == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained(model_path)
    else:
        raise ValueError(f'Technology {technology} is not supported.')

    #Load the model from the specified path
    model = torch.load(model_path)

    return model, tokenizer

@model_bp.route('/models', methods=['GET'])
def get_models():
    """
    Get the list of available models in the models directory.
    :return: JSON response containing the list of model files
    :rtype: JSON
    """
    #Directory containing the models
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models_GUI")
    #List of model files with '.pt' extension
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
    return jsonify(model_files)

@model_bp.route('/classify', methods=['POST'])
def classify_text():
    """
    Classify text using the pre-trained model.
    :return: JSON response containing the classification result
    :rtype: JSON
    """
    data = request.get_json()
    text = data['text']
    model_file = data['model_file']
    technology = data['technology']

    #Define the path to model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models_GUI")
    model_path = os.path.join(models_dir, model_file)

    #Load the model and tokenizer based on selected technology
    model, tokenizer = load_model_and_tokenizer(model_path, technology)

    #Use computational optimizer if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Encode input data
    encoding_text = tokenizer.encode_plus(
        text, #Original message
        max_length=current_app.config['configuration_main']['MAX_DATA_LEN'], #Maximum number of tokens (including special tokens)
        truncation=True, #Ignore tokens beyond set number of tokens
        add_special_tokens=True, #Add special tokens [CLS], [SEP], and [PAD]
        return_token_type_ids=False,
        padding='max_length', #Pad the input to the maximum length with [PAD] tokens
        return_attention_mask=True, #Create an attention mask to focus on non-padding tokens
        return_tensors='pt' #Return the encoded inputs as PyTorch tensors
    )

    input_ids = encoding_text['input_ids'].to(device) #Numeric input tokens and special tokens
    attention_mask = encoding_text['attention_mask'].to(device) #Attention mask

    #Compute the model outputs
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    #Calculate predictions. Take the maximum of 2 outputs
    #If the first one is the maximum, classify as "Change"; if the second one is the maximum, classify as "Non-Change"
    _, preds = torch.max(outputs, dim=1)

    result = "Change" if preds else "Non-Change"

    return jsonify({'classification': result})