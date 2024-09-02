"""File defining the neural network model class based on AlBERT"""

from torch import nn
from transformers import AlbertModel


class AlBERTSentimentClassifier(nn.Module):
    """
    Class implementing a AlBERT classification model with all its layers.
    """

    def __init__(self, number_classes, pre_trained_model_name, drop_out, transfer_learning):
        """
        Init function of AlBERTSentimentClassifier class
        :param number_classes: Number of final classification types an input text can have
        :type: Int
        :param pre_trained_model_name: Name of the selected AlBERT pre-trained model
        :type: String
        :param drop_out: Number per one of neurons that are deactivated in the network
        :type: Decimal
        :param transfer_learning: Indicates transfer learning application or complete BERT upgrade
        :type: Boolean
        :return: Nothing
        """

        #Initializer required for the model
        super(AlBERTSentimentClassifier,self).__init__()

        #AlBERT neural network model
        self.albert = AlbertModel.from_pretrained(pre_trained_model_name)

        #Application of transfer learning model alBERT
        if transfer_learning:
            for param in self.albert.parameters():
                param.requires_grad = False

        # Extra specific layer of neurons to avoid overfitting
        self.drop = nn.Dropout(p=drop_out)
        # Extra neuron layer for text classification
        self.relu = nn.ReLU()
        # Adding an additional linear layer
        self.linear_1 = nn.Linear(self.albert.config.hidden_size, 128)
        # Another linear layer for the final output
        self.linear_2 = nn.Linear(128, number_classes)

        #Ensure the newly added layers require gradients
        if transfer_learning:
            for param in self.drop.parameters():
                param.requires_grad = True
            for param in self.relu.parameters():
                param.requires_grad = True
            for param in self.linear_1.parameters():
                param.requires_grad = True
            for param in self.linear_2.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        """
        Function needed in Pytorch to specify order of layers
        :param input_ids: Representative identifiers of input data
        :type: Tensor
        :param attention_mask: Attention mask for transformers technology
        :type: Tensor
        :return: Output of complete model
        :type: Tensor
        """

        #Encoded input and the encoding of the classification token resulting from passing through AlBERT layer are obtained
        albert_output = self.albert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            return_dict = True
        )

        #Encoding of the classification token is passed as input data of the drop layer
        #This vector contains all the essence of input data
        drop_output = self.drop(albert_output['pooler_output'])
        # Pass through the first linear layer
        linear_1_output = self.linear_1(drop_output)
        # Apply ReLU activation
        relu_output = self.relu(linear_1_output)
        # Pass through the second linear layer
        output = self.linear_2(relu_output)

        return output