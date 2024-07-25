"""File defining the creation class of the dataset"""

import csv
import os
import torch

from mod_message.message import Message as message


class Dataset:
    """
    Class containing the complete dataset
    """

    def __init__(self):
        """
        Init function of Dataset class
        :return: Nothing
        """

        #Path from which raw data is read
        self.__files_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'files')
        self.__dataset = []
        self.__tokenizer = ''
        self.__max_len = 0


    def read_file(self, name_file):
        """
        Function to read SB Opinions file which contains raw datas
        :param name_file: Name of file read
        :type: String
        :return: Lines of file read
        :type: list[dict[String:String]
        """

        read_lines = []
        file_path = os.path.join(self.__files_path, f'{name_file}')

        with open(file_path, encoding='utf-8') as file:
            content = csv.DictReader(file)
            for line in content:
                read_lines.append(line)

        return read_lines


    def format_dataset(self, list_data):
        """
        Function to transform raw data into dataset object
        :param list_data: List with raw data of dataset
        :type: list[dict[String:String]]
        :return: Nothing
        """

        self.__dataset = list(map(lambda line: message(line), list_data))


    def __len__(self):
        """
        Function returning the number of data in the dataset
        :return: Number of data
        :type: Int
        """

        return len(self.__dataset)


    def __getitem__(self, item):
        """
        Function that returns a specific data from datasaet with its raw and encoded versions
        :param item: Number of the item to be recovered
        :type: Int
        :return: Dictionary with versions of data and its classification
        :type: dict[String:String,String:Tensor]
        """

        #Raw text and its classification are obtained
        text = self.__dataset[item].get_text()
        text_clasification = self.__dataset[item].get_type_class()

        #Coding of input data
        encoding = self.__tokenizer.encode_plus(
            text, #Original message
            max_length = self.__max_len, #Maximum number of tokens (counting special tokens)
            truncation = True, #Ignoring tokens beyond the set number of tokens
            add_special_tokens = True, #Special tokens [CLS], [SEP] and [PAD] added
            return_token_type_ids = False,
            padding = 'max_length', #If total number of tokens is less than the established maximum, it is filled with [PAD] until the maximum is reached
            return_attention_mask = True, #Model is instructed to pay attention only to non-empty tokens during training
            return_tensors = 'pt' #Final result of the encoder in Pytorch numbers
        )

        return {
            'text': text, #Raw input text
            'input_ids': encoding['input_ids'].flatten(), #Numeric input tokens and special tokens
            'attention_mask': encoding['attention_mask'].flatten(), #Attention mask
            'text_clasification': torch.tensor(text_clasification, dtype=torch.float) #Category to which the text belongs
        }


    def set_tokenizer(self, tokenizer):
        """
        Modify tokenizer value
        :param tokenizer: New tokenizer
        :type: Tokenizer
        :return: Nothing
        """

        self.__tokenizer = tokenizer


    def set_max_len(self,max_len):
        """
        Modify maximum number of tokens
        :param max_len: Maximum number of tokens to be set on the encoder
        :type: Int
        :return: Nothing
        """

        self.__max_len = max_len