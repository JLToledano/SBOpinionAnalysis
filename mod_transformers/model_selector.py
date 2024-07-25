"""File to select different models and prepare the appropriate configuration"""

from transformers import BertTokenizer, AlbertTokenizer, RobertaTokenizer
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.theme import Theme

from mod_transformers.mod_BERT.model_BERT import BERTSentimentClassifier
from mod_transformers.mod_alBERT.model_alBERT import AlBERTSentimentClassifier
from mod_transformers.mod_roBERTa.model_roBERTa import RoBERTaSentimentClassifier

def CLI_model_selector(configuration):
    """
    User chooses neural network model and its configuration is set up
    :param configuration: General model configurations
    :type: dict[String:String]
    :return: Basic Model configured (if applicable)
    :type: MODELSentimentClassifier
    :return: Function that transforms input data into special codes (tokens)
    :type: Tokenizer
    """

    custom_theme = Theme({"success":"green", "error":"red", "option":"yellow"})
    console = Console(theme = custom_theme)
    
    model_selection = False
    pre_trained_model_configurations = {
        'BERT': 'BERT_configurations(configuration)',
        'ALBERT': 'AlBERT_configurations(configuration)',
        'ROBERTA': 'ROBERTA_configurations(configuration)',
    }
    
    #As long as a pre-trained model has not been selected
    while not model_selection:
        console.print(Panel.fit("TECNOLOGÍAS DISPONIBLES"))
        #Available options are printed
        for pre_trained_model in pre_trained_model_configurations.keys():
            console.print("[option]" + pre_trained_model + "[/option]")

        console.print("")
        selected_option = Prompt.ask("Seleccione una opción")

        #User option is validated
        if selected_option in pre_trained_model_configurations.keys():
            model_selection = True
        else:
            console.print("[error]Opción incorrecta[/error], por favor seleccione una de las [success]opciones disponibles[/success].\n")

    #Configured model and tokenizer are returned.
    model_configuration = eval(pre_trained_model_configurations[selected_option])

    return model_configuration
    


def BERT_configurations(configuration):
    """
    Configuration of the BERT model and tokeniser
    :param configuration: General model configurations
    :type: dict[String:String]
    :return: BERT model and BERT tokenizer
    :type: dict[String:MODELSentimentClassifier/Tokenizer]
    """

    #Function transforming input data into special codes (tokens) for BERT model
    tokenizer = BertTokenizer.from_pretrained(configuration['PRE_TRAINED_MODEL_NAME']['BERT'])
    
    #Creation of BERT model
    model = BERTSentimentClassifier(configuration['NUM_TYPES_CLASSIFICATION_CLASSES'], configuration['PRE_TRAINED_MODEL_NAME']['BERT'], configuration['DROP_OUT_BERT'], configuration['TRANSFER_LEARNING'])

    BERT_configuration = {}
    BERT_configuration['model'] = model
    BERT_configuration['tokenizer'] = tokenizer
    BERT_configuration['name_model'] = 'BERT'

    return BERT_configuration


def AlBERT_configurations(configuration):
    """
    Configuration of the AlBERT model and tokeniser
    :param configuration: General model configurations
    :type: dict[String:String]
    :return: AlBERT model and AlBERT tokenizer
    :type: dict[String:MODELSentimentClassifier/Tokenizer]
    """

    #Function transforming input data into special codes (tokens) for AlBERT model
    tokenizer = AlbertTokenizer.from_pretrained(configuration['PRE_TRAINED_MODEL_NAME']['AlBERT'])
    
    #Creation of BERT model
    model = AlBERTSentimentClassifier(configuration['NUM_TYPES_CLASSIFICATION_CLASSES'], configuration['PRE_TRAINED_MODEL_NAME']['AlBERT'], configuration['DROP_OUT_BERT'], configuration['TRANSFER_LEARNING'])

    AlBERT_configuration = {}
    AlBERT_configuration['model'] = model
    AlBERT_configuration['tokenizer'] = tokenizer
    AlBERT_configuration['name_model'] = 'AlBERT'

    return AlBERT_configuration


def ROBERTA_configurations(configuration):
    """
    Configuration of the RoBERTa model and tokeniser
    :param configuration: General model configurations
    :type: dict[String:String]
    :return: RoBERTa model and RoBERTa tokenizer
    :type: dict[String:MODELSentimentClassifier/Tokenizer]
    """

    #Function transforming input data into special codes (tokens) for RoBERTa model
    tokenizer = RobertaTokenizer.from_pretrained(configuration['PRE_TRAINED_MODEL_NAME']['RoBERTa'])
    
    #Creation of RoBERTa model
    model = RoBERTaSentimentClassifier(configuration['NUM_TYPES_CLASSIFICATION_CLASSES'], configuration['PRE_TRAINED_MODEL_NAME']['RoBERTa'], configuration['DROP_OUT_BERT'], configuration['TRANSFER_LEARNING'])

    RoBERTa_configuration = {}
    RoBERTa_configuration['model'] = model
    RoBERTa_configuration['tokenizer'] = tokenizer
    RoBERTa_configuration['name_model'] = 'RoBERTa'

    return RoBERTa_configuration