"""File defining the creation class of each data on dataset"""

class Message:
    """Class containing one case of dataset"""

    def __init__(self, dict_data):
        """
        Init function of Message class
        :param dict_data: One line of raw data with text and classification (class)
        :type: dict[String:String]
        :return: Nothing
        """

        self.__text = dict_data['text']

        #Written classification is translated into a numerical classification.
        if (dict_data['class'] == 'Change'):
            self.__type_class = 1 #Change
        else:
            self.__type_class = 0 #Non-Change


    def get_text(self):
        """
        Obtaining input data message
        :return: Message from test subject
        :type: String
        """

        return self.__text


    def get_type_class(self):
        """
        Obtaining message classification
        :return: Classification messague from test subject
        :type: Int
        """

        return self.__type_class