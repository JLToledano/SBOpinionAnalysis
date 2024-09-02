"""File defining the neural network model class based on BERT"""

from torch import nn
from transformers import BertModel


class BERTSentimentClassifier(nn.Module):
    """
    Class implementing a BERT classification model with all its layers.
    """

    def __init__(self, number_classes, pre_trained_model_name, drop_out, transfer_learning):
        """
        Init function of BERTSentimentClassifier class
        :param number_classes: Number of final classification types an input text can have
        :param pre_trained_model_name: Name of the selected BERT pre-trained model
        :param drop_out: Number of neurons to deactivate in the network
        :param transfer_learning: Indicates transfer learning application or complete BERT upgrade
        """
        super(BERTSentimentClassifier, self).__init__()

        # BERT neural network model
        self.bert = BertModel.from_pretrained(pre_trained_model_name)

        # Application of transfer learning model BERT
        if transfer_learning:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Extra specific layer of neurons to avoid overfitting
        self.drop = nn.Dropout(p=drop_out)
        # Extra neuron layer for text classification
        self.relu = nn.ReLU()
        # Adding an additional linear layer
        self.linear_1 = nn.Linear(self.bert.config.hidden_size, 128)
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
        :param attention_mask: Attention mask for transformers technology
        :return: Output of the complete model
        """
        # Pass through BERT
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Apply dropout to the pooled output
        drop_output = self.drop(bert_output['pooler_output'])
        # Pass through the first linear layer
        linear_1_output = self.linear_1(drop_output)
        # Apply ReLU activation
        relu_output = self.relu(linear_1_output)
        # Pass through the second linear layer
        output = self.linear_2(relu_output)

        return output