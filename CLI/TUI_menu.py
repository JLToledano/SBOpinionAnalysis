"""File containing the console message printing functions"""

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.align import Align
from rich.panel import Panel
from rich.theme import Theme
from rich.table import Table, Column
from rich.prompt import Prompt, IntPrompt, FloatPrompt
from rich.tree import Tree


def welcome_menu():
    """
    Function which prints the welcome message when starting the application
    :return: Nothing
    """

    console = Console()

    SB_message = """
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m- - - - - - - - -   - - - - - - - - -m- - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m-               -   -     - - - -   -m- - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m-   - - - - - - -   -     -     -   -m- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m-   -               -     - - - -   -m- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m-   - - - - - - -   -   - - - - - - -m- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m-               -   -   - - - - - - -m- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m- - - - - - -   -   -     - - - -   -m- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m            -   -   -     -     -   -m- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m- - - - - - -   -   -     - - - -   -m- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - m- - - - - - - - -   - - - - - - - - -m- - - - - - - - - - - - - - - - - - - - - - - - - - - -
    """

    opinion_message = """
    - - - - - - - - m- - - - - - - - -   - - - - - - - - -   - - - - - - - - -   - -         - - -   - - - - - - - - -   - - - - - - - - -   - -         - - -   - - - - - - - - -m- - - - - - - -
    - - - - - - - - m-               -   -     - - - -   -   -               -   -   -       -   -   -               -   -               -   -   -       -   -   -               -m- - - - - - - -
    - - - - - - - - m-               -   -     -     -   -   - - - -   - - - -   -     -     -   -   - - - -   - - - -   -               -   -     -     -   -   -   - - - - - - -m- - - - - - - -
    - - - - - - - - m-   - - - - -   -   -     -     -   -         -   -         -       -   -   -         -   -         -   - - - - -   -   -       -   -   -   -   -            m- - - - - - - -
    - - - - - - - - m-   -       -   -   -     - - - -   -         -   -         -   -     - -   -         -   -         -   -       -   -   -   -     - -   -   -   - - - - - - -m- - - - - - - -
    - - - - - - - - m-   -       -   -   -     - - - - - -         -   -         -   - -     -   -         -   -         -   -       -   -   -   - -     -   -   -               -m- - - - - - - -
    - - - - - - - - m-   - - - - -   -   -     -                   -   -         -   -   -       -         -   -         -   - - - - -   -   -   -   -       -   - - - - - - -   -m- - - - - - - -
    - - - - - - - - m-               -   -     -             - - - -   - - - -   -   -     -     -   - - - -   - - - -   -               -   -   -     -     -               -   -m- - - - - - - -
    - - - - - - - - m-               -   -     -             -               -   -   -       -   -   -               -   -               -   -   -       -   -   - - - - - - -   -m- - - - - - - -
    - - - - - - - - m- - - - - - - - -   - - - -             - - - - - - - - -   - - -         - -   - - - - - - - - -   - - - - - - - - -   - - -         - -   - - - - - - - - -m- - - - - - - -
    """

    analysis_message = """
    - - - - - - - - m      - - -         - -         - - -         - - -         - - -               - - - -   - - - -   - - - - - - - - -   - - - - - - - - -   - - - - - - - - -m- - - - - - - -
    - - - - - - - - m    -       -       -   -       -   -       -       -       -   -               -     -   -     -   -               -   -               -   -               -m- - - - - - - -
    - - - - - - - - m  -     -     -     -     -     -   -     -     -     -     -   -               -     -   -     -   -   - - - - - - -   - - - -   - - - -   -   - - - - - - -m- - - - - - - -
    - - - - - - - - m-     -   -     -   -       -   -   -   -     -   -     -   -   -               -     - - -     -   -   -                     -   -         -   -            m- - - - - - - -
    - - - - - - - - m-     - - -     -   -   -     - -   -   -     - - -     -   -   -               -               -   -   - - - - - - -         -   -         -   - - - - - - -m- - - - - - - -
    - - - - - - - - m-               -   -   - -     -   -   -               -   -   -               - - - -   - - - -   -               -         -   -         -               -m- - - - - - - -
    - - - - - - - - m-   - - - - -   -   -   -   -       -   -   - - - - -   -   -   -                     -   -         - - - - - - -   -         -   -         - - - - - - -   -m- - - - - - - -
    - - - - - - - - m-   -       -   -   -   -     -     -   -   -       -   -   -   - - - - - - -         -   -                     -   -   - - - -   - - - -               -   -m- - - - - - - -
    - - - - - - - - m-   -       -   -   -   -       -   -   -   -       -   -   -               -         -   -         - - - - - - -   -   -               -   - - - - - - -   -m- - - - - - - -
    - - - - - - - - m- - -       - - -   - - -         - -   - - -       - - -   - - - - - - - - -         - - -         - - - - - - - - -   - - - - - - - - -   - - - - - - - - -m- - - - - - - -
    """


    SB_message_align = Align.center(SB_message, vertical="middle")
    opinion_message_align = Align.center(opinion_message, vertical="middle")
    analysis_message_align = Align.center(analysis_message, vertical="middle")


    console.print(SB_message_align, style="bold blue")
    console.print(opinion_message_align, style="bold blue")
    console.print(analysis_message_align, style="bold blue")


    welcome_message= """# Bienvenido a la aplicación"""
    welcome_message_markdown = Markdown(welcome_message)
    console.print(welcome_message_markdown)


def sample_data_menu(number_train_data, number_test_data):
    """
    Function which prints amount of data present for training and evaluation
    :param number_train_data:Number of data dedicated to training
    :type: Int
    :param number_test_data: Number of data dedicated to testing
    :type: Int
    :return: Nothing
    """
    
    console = Console()

    #Training data panel
    training_data_message = "DATOS DE ENTRENAMIENTO: " + str(number_train_data)
    training_data_message_align = Align(training_data_message, align="left")
    console.print(Panel(training_data_message_align, style="bold"))

    #Evaluation data panel
    evaluation_data_message = "DATOS DE EVALUACIÓN: " + str(number_test_data)
    evaluation_data_message_align = Align(evaluation_data_message, align="left")
    console.print(Panel(evaluation_data_message_align, style="bold"))

def option_menu():
    """
    Function which prints the options message
    :return: Nothing
    """

    console = Console()

    option_message = """Escoge una de las siguientes opciones"""
    option_message_align = Align(option_message, align="left")
    console.print(Panel.fit(option_message_align, style="bold"))

    first_option_message = """1. Utilizar modelo preentrenado"""
    first_option_message_markdown = Markdown(first_option_message)
    console.print(first_option_message_markdown, style="bold")

    second_option_message = """2. Entrenar modelo"""
    second_option_message_markdown = Markdown(second_option_message)
    console.print(second_option_message_markdown, style="bold")

    third_option_message = """3. Evaluar modelo"""
    third_option_message_markdown = Markdown(third_option_message)
    console.print(third_option_message_markdown, style="bold")

    fourth_option_message = """4. Personalizar configuración de parámetros"""
    fourth_option_message_markdown = Markdown(fourth_option_message)
    console.print(fourth_option_message_markdown, style="bold")

    fifth_option_message = """5. Personalizar fichero dataset"""
    fifth_option_message_markdown = Markdown(fifth_option_message)
    console.print(fifth_option_message_markdown, style="bold")

    sixth_option_message = """6. Ayuda"""
    sixth_option_message_markdown = Markdown(sixth_option_message)
    console.print(sixth_option_message_markdown, style="bold")

    seventh_option_message = """7. Salir"""
    seventh_option_message_markdown = Markdown(seventh_option_message)
    console.print(seventh_option_message_markdown, style="bold")

    console.print("")


def help_menu():
    """
    Function which prints help messages
    :return: Nothing
    """
    
    #Customization of the console with special predefined styles
    custom_theme = Theme({"success":"green", "error":"red", "option":"yellow", "required_parameter":"purple"})
    console = Console(theme = custom_theme)

    #Design and printing of welcome message
    welcome_message= """# Bienvenido a la aplicación. Gracias a esta herramienta podrás utilizar modelos de Inteligencia Artificial para analizar la opiniones de usuario de un Smart Building"""
    welcome_message_markdown = Markdown(welcome_message)
    console.print(welcome_message_markdown)

    #Design and printing of the liability warning message
    console.print(Panel.fit("[error]WARNING[/error]", border_style="red"))
    console.print("""[error]Los resultados que pueda reflejar esta aplicación solamente deben ser tomados en consideración por especialistas técnicos junto con pruebas externas que reflejen si es necesario tomar la opinión en consideración.[/error]""")
    console.print(Panel.fit("[error]WARNING[/error]", border_style="red"))
    console.print()

    #Design and printing of the explanatory message menu option 1
    first_option_table = Table(expand = True)
    first_option_table.add_column("[option]OPCIÓN 1[/option] Utilizar modelo preentrenado", justify="full")
    first_option_table.add_row(
    """En esta opción se podrá elegir una de las redes neuronales pre-entrenadas disponibles en la aplicación. Después de efectuar la elección, se pedirá un texto que deberá escribirse en inglés y el cuál representa un mensaje que será analizado por la red neuronal. El resultado del análisis reflejará si el mensaje tiene señales de que las condiciones del edificio deben de ser modificadas o no"""
    )
    console.print(first_option_table)

    #Design and printing of the explanatory message menu option 2
    second_option_table = Table(expand = True)
    second_option_table.add_column("[option]OPCIÓN 2[/option] Entrenar modelo", justify = "full")
    second_option_table.add_row(
    """En esta opción se podrá elegir una base pre-entrenada como modelo de red neronal a la que se le añaden capas extras para su adaptación al uso dado en esta aplicación. Después de la elección se entrenará el modelo con el dataset proporcionado y se almacenará el modelo entrenado en un fichero con el nombre especificado por el usuario para su posible uso posterior.\n"""
    + """\n""" +
    """Por defecto el dataset proporcionado contiene datos que se destinarán a entrenamiento y datos que se destinarán a evaluación."""
    )
    console.print(second_option_table)

    #Design and printing of the explanatory message menu option 3
    third_option_table = Table(expand = True)
    third_option_table.add_column("[option]OPCIÓN 3[/option] Evaluar modelo", justify = "full")
    third_option_table.add_row(
    """En esta opción se podrá elegir un modelo pre-entrenado disponible en la aplicación para evaluar su rendimiento con el dataset de datos proporcionado.\n"""
    + """\n""" +
    """Por defecto el dataset proporcionado contiene datos que se destinarán a entrenamiento y datos que se destinarán a evaluación."""
    )
    console.print(third_option_table)

    #Design and printing of the explanatory message menu option 4
    fourth_option_table_parameters = Table(box = box.HEAVY)
    fourth_option_table_parameters.add_column("Parámetro", justify = "center")
    fourth_option_table_parameters.add_column("Valor por defecto", justify = "center")
    fourth_option_table_parameters.add_column("Definición", justify = "full")
    fourth_option_table_parameters.add_column("Consideraciones", justify = "center")
    
    fourth_option_table_parameters.add_row("MAX_DATA_LEN", "200", "Número de palabras máximo que admite el modelo neuronal. El resto de palabras se truncan", "")
    fourth_option_table_parameters.add_row("BATCH_SIZE", "16", "Tamaño de lote. Número de datos que se insertan en la red neuronal cada vez", "")
    fourth_option_table_parameters.add_row("EPOCHS", "3", "Número de épocas (iteraciones)", "")
    fourth_option_table_parameters.add_row("DATALOADER_NUM_WORKERS", "1", "Número de procesos que se ejecutan en paralelo. Se analizan X datos en paralelo", "")
    fourth_option_table_parameters.add_row("TRANSFER_LEARNING", "True", "Indicación de si se aprovechan los conocimientos de la tecnología base o se entrena el modelo completo", "")
    fourth_option_table_parameters.add_row("DROP_OUT_BERT", "0.3", "Tanto por uno de neuronas que se desactivan aleatoriamente en una de las capas de la red nueronal", "[option]APLICABLE EN BERT[/option]")

    fourth_option_table = Table(expand = True)
    fourth_option_table.add_column("[option]OPCIÓN 4[/option] Personalizar configuración de parámetros", justify = "full")
    fourth_option_table.add_row(
    """En esta opción el usuario podrá cambiar la configuración inicial de los parámetros de división del dataset de datos, entrenamiento y evaluación.\n"""
    + """\n""" +
    """Los cambios realizados perduran mientras se esté ejecutando la aplicación. El reinicio de esta significa el reseteo de los cambios realizados.\n"""
    + """\n""" +
    """La configuración inicial es la siguiente:\n"""
    )
    fourth_option_table.add_row(fourth_option_table_parameters)
    console.print(fourth_option_table)

    #Design and printing of the explanatory message menu option 5
    fifth_option_table = Table(expand = True)
    fifth_option_table.add_column("[option]OPCIÓN 5[/option] Personalizar fichero dataset", justify = "full")
    fifth_option_table.add_row(
    """En esta opción el usuario puede cambiar la ruta predefinida del dataset para utilizar uno configurado por el usuario pero que esté situado en el mismo directorio destinado a datasets de la aplicación.\n"""
    + """\n""" +
    """La extensión del archivo deberá ser .csv y con el siguiente formato de mensajes:\n"""
    + """\n""" +
    """             [required_parameter]"Texto a analizar entre comillas dobles"[/required_parameter],[required_parameter]clasificación[/required_parameter]\n"""
    + """\n""" +
    """La clasificación deberá ser: [option]Change[/option] o [option]Non-Change[/option]. El texto a analizar puede tener varios párrafos separados por retorno de carro."""
    )
    console.print(fifth_option_table)

    #Design and printing of the explanatory message menu option 6
    sixth_option_table = Table(expand = True)
    sixth_option_table.add_column("[option]OPCIÓN 6[/option] Ayuda", justify = "full")
    sixth_option_table.add_row(
    """Con esta opción se imprime por pantalla la guía de ayuda al usuario de la aplicación. La guía que se muestra en este momento es la que está disponible.\n"""
    + """\n""" +
    """Para más información, acudid a la documentación externa asociada al aplicativo."""
    )
    console.print(sixth_option_table)

    #Design and printing of the explanatory message menu option 7
    seventh_option_table = Table(expand = True)
    seventh_option_table.add_column("[option]OPCIÓN 7[/option] Salir", justify = "full")
    seventh_option_table.add_row(
    """Con esta opción finaliza la ejecución de la aplicación.\n"""
    + """\n""" +
    """Los modelos entrenados persistirán al apagado de la aplicación pero no los cambios realizados en la configuración de entrenamiento."""
    )
    console.print(seventh_option_table)


def metrics_menu(confusion_matrix, accurancy, recall, precision, f1, execution_time):
    """
    Function that prints out the results of the metrics from a training or an evaluation
    :param confusion_matrix: Confusion matrix result value
    :type: Array[Array[Int]]
    :param accurancy: Accurancy result value
    :type: Float
    :param recall: Recall result value
    :type: Float
    :param precision: Precision result value
    :type: Float
    :param f1: F1 result value
    :type: Float
    :param execution_time: Execution training or evaluation time
    :type: Float
    :return: Nothing
    """

    #Customization of the console with special predefined styles
    custom_theme = Theme({"parameter":"purple"})
    console = Console(theme = custom_theme)

    #Section header design
    section_title_message = "RESULTADOS MEDIDAS"
    section_title_message_align = Align(section_title_message, align="center")
    console.print(Panel(section_title_message_align, style="bold"))

    #Execution time panel
    execution_time_message = "TIEMPO DE EJECUCIÓN: " + str(execution_time) + " segundos"
    execution_time_message_align = Align(execution_time_message, align="left")
    console.print(Panel(execution_time_message_align, style="bold"))

    #Confusion Matrix parts
    true_negative_panel = Panel(Align("[parameter]TN[/parameter] " + str(confusion_matrix[0][0]), align="center"), title="True Negative")
    false_positive_panel = Panel(Align("[parameter]FP[/parameter] " + str(confusion_matrix[0][1]), align="center"), title="False Positive")
    false_negative_panel = Panel(Align("[parameter]FN[/parameter] " + str(confusion_matrix[1][0]), align="center"), title="False Negative")
    true_positive_panel = Panel(Align("[parameter]TP[/parameter] " + str(confusion_matrix[1][1]), align="center"), title="True Positive")

    #Accurancy panel
    accurancy_panel = Panel(Align(str(accurancy), align="center"), title="Accurancy")

    #Recall panel
    recall_panel = Panel(Align(str(recall), align="center"), title="Recall")

    #Precision panel
    precision_panel = Panel(Align(str(precision), align="center"), title="Precision")

    #F1 panel
    f1_panel = Panel(Align(str(f1), align="center"), title="F1")

    #General table design
    measurements_table = Table(
        Column(header="Matriz de Confusión", justify="center"),
        Column(header="Otras Medidas", justify="center"),
        expand=True
    )

    #Design of confusion matrix table (left side general table)
    confusion_matrix_table = Table.grid(expand=True)
    confusion_matrix_table.add_column(justify="center")
    confusion_matrix_table.add_column(justify="center")
    confusion_matrix_table.add_row(true_negative_panel,false_positive_panel)
    confusion_matrix_table.add_row(false_negative_panel,true_positive_panel)

    #Design of table of other measurements (right side of general table)
    other_measurements_table = Table.grid(expand=True)
    other_measurements_table.add_column(justify="center")
    other_measurements_table.add_column(justify="center")
    other_measurements_table.add_row(accurancy_panel,recall_panel)
    other_measurements_table.add_row(precision_panel,f1_panel)

    #Connection of subtables with the general table
    measurements_table.add_row(confusion_matrix_table,other_measurements_table)

    #The set of tables is printed
    console.print(measurements_table)
    console.print('\n')


def models_menu(list_models_files):
    """
    Function that simulates a pagination of the list of models and allows the user to select one of them 
    for use or evaluation
    :param list_models_files: List of all available model files
    :type: List[String]
    :return: Name of selected file
    :type: String
    """

    custom_theme = Theme({"error":"red", "range":"yellow", "index":"purple"})
    console = Console(theme = custom_theme)

    #User is asked how many files he/she wants for each directory in the "paging"
    dir_number_files = IntPrompt.ask("Por favor indique el número de ficheros por directorio", default=25)
    #If you select either files per directory or a negative quantity, one file per directory is assigned by default
    if dir_number_files <= 0:
        dir_number_files = 1

    #Resulting number of directories is calculated
    number_dirs = len(list_models_files) // dir_number_files
    #If you select more files per directory than available files, at least one directory with all files is set to be available
    #Or if division is not exact, one more directory is added 
    if number_dirs == 0 or (len(list_models_files) % dir_number_files):
        number_dirs += 1

    #Instance of the directory tree
    dir_tree = Tree("Directorios")

    #For each directory
    for i in range(1, number_dirs + 1):
        #If we are in last directory, it loads from corresponding file to last one
        #It may be that the number of files is not an exact multiple of the dividing number
        if i == number_dirs:
            #Directory name is composed
            title_dir = "[index]" + str(i) + "[/index]" + ". " + "Ficheros del " + "[range]" + str((i - 1) * dir_number_files + 1) + "[/range]" + " al " + "[range]" + str(len(list_models_files)) + "[/range]"

        else:
            #Directory name is composed
            title_dir = "[index]" + str(i) + "[/index]" + ". " + "Ficheros del " + "[range]" + str((i - 1) * dir_number_files + 1) + "[/range]" + " al " + "[range]" + str(i * dir_number_files) + "[/range]"

        #Directory is added to the directory tree
        dir_tree.add(title_dir)

    #Directory tree is printed
    console.print(dir_tree)
    console.print("\n")

    selected_file = False

    #As long as no file has been selected
    while not selected_file:
        #Instance of the directory tree
        dir_tree = Tree("Directorios")

        #User is asked if he/she wants to display directory or select file
        selection = Prompt.ask("¿Qué desea seleccionar?", choices=["directorio", "fichero"])

        #If he/she wants to display a directory
        if selection == "directorio":
            #User is asked which directory to display
            number_dir_displayed = IntPrompt.ask("Por favor indique el número de directorio a desplegar", default=1)

            #If user has chosen an existing directory
            if 1 <= number_dir_displayed <= number_dirs:
                #Directory tree is composed with existing directories and chosen directory displayed
                for i in range(1, number_dirs + 1):
                    #If we are in last directory, it loads from corresponding file to last one
                    #It may be that the number of files is not an exact multiple of the dividing number
                    if i == number_dirs:
                        #Directory name is composed
                        title_dir = "[index]" + str(i) + "[/index]" + ". " + "Ficheros del " + "[range]" + str((i - 1) * dir_number_files + 1) + "[/range]" + " al " + "[range]" + str(len(list_models_files)) + "[/range]"
                        #If it is the directory chosen to deploy it
                        if i == number_dir_displayed:
                            #Position in list of first file belonging to this directory is set
                            minimum_of_range = (i - 1) * dir_number_files
                            #Position in list of last file belonging to this directory is set
                            #As last directory, it is last file in list
                            maximum_of_range = len(list_models_files)

                    else:
                        #Directory name is composed
                        title_dir = "[index]" + str(i) + "[/index]" + ". " + "Ficheros del " + "[range]" + str((i - 1) * dir_number_files + 1) + "[/range]" + " al " + "[range]" + str(i * dir_number_files) + "[/range]"
                        #If it is the directory chosen to deploy it
                        if i == number_dir_displayed:
                            #Position in list of first file belonging to this directory is set
                            minimum_of_range = (i - 1) * dir_number_files
                            #Position in list of last file belonging to this directory is set
                            maximum_of_range = i * dir_number_files

                    #If it is the directory chosen to deploy it
                    if i == number_dir_displayed:
                        #If there is only one file in directory, maximum range is increased by one 
                        #because in Python if minimum and maximum range coincide an empty list is returned
                        if minimum_of_range == maximum_of_range:
                            maximum_of_range += 1

                        #Files that belong to directory to be deployed are obtained
                        list_print_files = list_models_files[minimum_of_range:maximum_of_range]

                        #Directory is added to the directory tree
                        displayed_branch = dir_tree.add(title_dir)

                        #Ratio of files to total number of files in list
                        number_file = (i - 1) * dir_number_files + 1

                        #For each file belonging to directory
                        for file in list_print_files:
                            #File name is composed
                            title_file = "[index]" + str(number_file) + "[/index]" + ". " + file
                            #File is added to the displayed directory tree
                            displayed_branch.add(title_file)

                            #File index is increased
                            number_file += 1

                    #If it is not the directory chosen to deploy it
                    else:
                        dir_tree.add(title_dir)

                #Directory tree is printed
                console.print(dir_tree)
                console.print("\n")

            #If user has chosen a directory that does not exist
            else:
                console.print("Número de [error]directorio no válido[/error]\n")

        #If he/she wants to select a file
        else:
            #User is asked which directory to select
            file_position = IntPrompt.ask("Por favor indique el número de fichero que desea seleccionar", default=1)

            #If position of chosen file exists, file name is obtained and returned
            try:
                #This filter is performed because the position 0 - 1 (-1) in Python returns last value in list. 
                #In addition, negative values are not valid
                if file_position <= 0:
                    console.print("Número de [error]fichero no válido[/error]\n")

                else:
                    #File name is obtained
                    file_name = list_models_files[file_position - 1]

                    #It indicates that a valid file has already been selected
                    selected_file = True
            
            #If file position does not exist, it indicates that selection is invalid
            except:
                console.print("Número de [error]fichero no válido[/error]\n")

    return file_name