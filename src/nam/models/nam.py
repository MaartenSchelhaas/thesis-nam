"""
NAM — Neural Additive Model.

Each input feature gets its own FeatureNN subnet, learned functions f_i(x_i)
are summed to produce the final prediction:

    y = b + f_0(x_0) + f_1(x_1) + ... + f_n(x_n)
"""

import torch
import torch.nn as nn
from .feature_nn import FeatureNN
from ..utils.config import NAMConfig


class NAM(nn.Module):
    """
    Neural Additive Model: one FeatureNN per input feature.

    Args:
        num_features: Number of input features (= number of FeatureNNs to create).
        config:       NAMConfig dataclass with all hyperparameters.
    """

    def __init__(
        self,
        num_features: int,
        num_units: int, #TODO: int for now, reference uses a list. Dont see a use for that though
        hidden_sizes: list,
        dropout: float,
        feature_dropout: float,
        activation: str,
    ):
        """Initialization of the Neural Additive Model class. 

        Args:
            num_features (int): Amount of input features
            num_units (int): Width of the activation layer of each subnet
            hidden_sizes (list): List of hidden layer widths after the activation layer.
            dropout (float): Dropout probability applied after each hidden layer in the subnets.
            feature_dropout (float): Probability of dropping an entire feature output before summation.
            activation (str): Activation layer type, 'exu' or 'relu'.
        """
        super().__init__()
        self.num_features = num_features


        self.feature_nns = nn.ModuleList([
            FeatureNN(
                num_units=num_units,
                hidden_sizes=hidden_sizes,
                dropout = dropout,
                activation=activation,
            )
            for _ in range(num_features)
        ])

        self.dropout_layer =  nn.Dropout(p=feature_dropout)
        self._bias = nn.Parameter(data=torch.zeros(1))
    
    def calc_outputs(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        """Pass each feature column through its dedicated FeatureNN subnet.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            list[torch.Tensor]: List of num_features tensors, each of shape (batch_size, 1), 
                                representing the learned contribution f_i(x_i) for each feature i.
        """
        individual_outputs = []
        for i in range(self.num_features):
            feature_input = inputs[:,i]
            feature_output = self.feature_nns[i](feature_input)
            individual_outputs.append(feature_output)
        return individual_outputs


    def forward(self, x: torch.Tensor):
        """Forward pass of the NAM. Each column of x gets passed into a seperate subnet, on which we apply
        feature dropout per observation, and we return the final prediction + bias term. 

        Args:
            x (torch.Tensor): Input data, of size (batch_size, num_features)

        Returns:
            _type_: Returns the final prediction, plus the individual contributions. 
        """
        # x shape: 
        # TODO: for each feature i, slice x[:, i:i+1] and pass through feature_nns[i]
        #       result per feature: (batch_size, 1)

        #Pass the respective columns to each feature subnet
        individual_outputs = self.calc_outputs(x)
        #Concenate indivdual tensors back toghether
        conc_out = torch.cat(individual_outputs, dim=-1)
        #Feature dropout per observation
        dropout_out = self.dropout_layer(conc_out)
        out = torch.sum(dropout_out, dim=-1)

        return out + self._bias, dropout_out

