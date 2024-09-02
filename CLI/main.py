"""Main application file. Contains the general operating logic"""

import __init__ as init
import numpy
import torch
import os
import sys

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)

from sklearn.model_selection import train_test_split
from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.prompt import Prompt

from TUI_menu import option_menu, welcome_menu, help_menu, sample_data_menu
from mod_dataset.dataset import Dataset
from CLI.menu_options import *

#Load constants and predefined application parameters
configuration_main = init.load_config_mentalapp()


def dataset_initialize():
    """
    Initialisation of the default dataset
    :return: Complete dataset, Training dataset, Evaluation dataset with correct format
    :type: Dataset
    """

    #Dataset with all data
    complete_dataset = Dataset()
    #Dataset with data for training
    train_dataset = Dataset()
    #Dataset with data for evaluation
    test_dataset = Dataset()

    #All data are read from the source file
    raw_data = complete_dataset.read_file(configuration_main['FILE_DATASET_NAME'])
    #A sample of the data is taken for testing.
    #raw_data =  raw_data[0:len(raw_data)//2]
    #The data is divided between training and evaluation. The parameter test_size marks the percent per one of data for evaluation.
    train_raw_data,test_raw_data = train_test_split(raw_data, test_size = 0.2, random_state = configuration_main['RANDOM_SEED'])
    sample_data_menu(len(train_raw_data),len(test_raw_data))

    #Dataframes are put in the required format.
    complete_dataset.format_dataset(raw_data)
    train_dataset.format_dataset(train_raw_data)
    test_dataset.format_dataset(test_raw_data)

    return complete_dataset, train_dataset, test_dataset


def main():
    """Main function of the application. Manages the menu and calls to the main blocks
    :return: Nothing
    """

    welcome_menu()
    
    #Random initialization of the weights and parameters of Pytorch model
    numpy.random.seed(configuration_main['RANDOM_SEED'])
    torch.manual_seed(configuration_main['RANDOM_SEED'])

    #Use of the computational optimizer if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #Raw dataset initialization
    complete_dataset,train_dataset,test_dataset = dataset_initialize()

    #Options available to the user
    selected_option = 0
    menu_options = {
        '1': "use_classify_model(configuration_main, device)",
        '2': "training_model_scratch(configuration_main, device, train_dataset, test_dataset)",
        '3': "evaluating_model_pretraining(configuration_main, device, test_dataset)",
        '4': "configuration_main = customize_parameter_configuration(configuration_main)",
        '5': "configuration_main['FILE_DATASET_NAME'] = assign_new_dataset()",
        '6': "help_menu()",
    }
    
    #As long as user does not select the exit option, program continues to run
    while selected_option != "7":
        option_menu()

        custom_theme = Theme({"success":"green", "error":"red", "option":"yellow", "required_parameter":"purple"})
        console = Console(theme = custom_theme)
        selected_option = Prompt.ask("Seleccione una opción")
        console.print(Panel.fit("Opción: " + selected_option))

        #User option is executed if possible
        if selected_option in menu_options.keys():
            #If execution fails, user is prompted to review parameter or application settings.
            try:
                exec(menu_options[selected_option])
                #If new dataset is specified, training and evaluation datasets are reloaded
                if selected_option == '5':
                    valid_file = False
                    while not valid_file:
                        #User is informed of result of data upload and, if necessary, is asked again for a file name
                        try:
                            complete_dataset,train_dataset,test_dataset = dataset_initialize()
                            console.print("[success]Nuevo fichero cargado[/success]\n", style="bold")
                            valid_file = True
                        except:
                            console.print("[error]Nombre de fichero incorrecto[/error]", style="bold")
                            console.print("Compruebe el [required_parameter]nombre de fichero[/required_parameter] introducido y si está situado en la [required_parameter]ruta[/required_parameter] [option]mentalapp/mod_dataset/files[/option] dentro del proyecto\n", style="bold")

                            exec(menu_options[selected_option])
            except:
                console.print("[error]No se ha podido ejecutar[/error] la opción elegida, por favor compruebe la configuración elegida e [success]inténtelo de nuevo[/success].")

        else:
            if selected_option == '7':
                pass
            else:
                console.print("[error]Opción incorrecta[/error], por favor seleccione una de las [success]opciones disponibles[/success].")
                print("")


if __name__ == "__main__":
    main()