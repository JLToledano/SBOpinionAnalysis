"""Initial configuration of the main application module"""
import os
import yaml


def load_config_mentalapp() -> dict:
    """
    Function that loads the general configuration of the application
    :return: Global constants of the application
    :rtype: dict[String:String]
    """
    
    configuration_path = os.path.dirname(__file__)
    with open(os.path.join(configuration_path, 'config.yaml'), 'r', encoding='utf-8') as file:
        info_config = yaml.safe_load(file)

    return info_config