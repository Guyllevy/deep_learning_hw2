import torch
import torch.nn as nn
import itertools as it
from torch import Tensor
from typing import Sequence

from .mlp import MLP, ACTIVATIONS, ACTIVATION_DEFAULT_KWARGS

POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class CNN(nn.Module):
    """
    A simple convolutional neural network model based on PyTorch nn.Modules.

    Has a convolutional part at the beginning and an MLP at the end.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.mlp = self._make_mlp()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        N = len(self.channels)
        P = self.pool_every
        
        self.non_lin_class = ACTIVATIONS[self.activation_type]
        pooling_class = POOLINGS[self.pooling_type]
        
        all_channels = [in_channels] + self.channels
        
        for i in range(N//P):
            # add layers: (CONV -> ACT)*P -> POOL
            for j in range(P):
                # add layers: CONV, ACT
                layers.append(nn.Conv2d(all_channels[j + i*P], all_channels[j + i*P + 1], **self.conv_params))
                layers.append(self.non_lin_class(**self.activation_params))
            # add POOL
            layers.append(pooling_class(**self.pooling_params))
            
        for k in range(N%P): # add leftover layers without pooling
            # add layers: CONV, ACT
            layers.append(nn.Conv2d(all_channels[P*(N//P) + k], all_channels[P*(N//P) + k + 1], **self.conv_params))
            layers.append(self.non_lin_class(**self.activation_params))
            
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:
            # ====== YOUR CODE: ======
            feature_extractor_output_shape = self.feature_extractor(torch.zeros((1,*self.in_size))).shape
            
            n_features = 1
            for dim in feature_extractor_output_shape:
                n_features *= dim
                
            return n_features
            # ========================
        finally:
            torch.set_rng_state(rng_state)

    def _make_mlp(self):
        # TODO:
        #  - Create the MLP part of the model: (FC -> ACT)*M -> Linear
        #  - Use the the MLP implementation from Part 1.
        #  - The first Linear layer should have an input dim of equal to the number of
        #    convolutional features extracted by the convolutional layers.
        #  - The last Linear layer should have an output dim of out_classes.
        mlp: MLP = None
        # ====== YOUR CODE: ======
        M = len(self.hidden_dims)
        dims = self.hidden_dims + [self.out_classes]
        nonlins = []
        for i in range(M):
            nonlins.append(self.non_lin_class(**self.activation_params))
        nonlins.append(nn.Identity())
            
        mlp = MLP(in_dim = self._n_features(), dims = dims, nonlins = nonlins)
        # ========================
        return mlp

    def forward(self, x: Tensor):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        out: Tensor = None
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x).flatten(start_dim = 1)
        out = self.mlp(features)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        
        non_lin_class = ACTIVATIONS[activation_type]
            
        all_channels = [in_channels] + channels
            
        main_layers = []
        
        if len(channels) == 1:
            main_layers.append(nn.Conv2d(all_channels[0], all_channels[1], kernel_sizes[0], stride = 1,padding = kernel_sizes[0]//2))
            
        else:
            for i in range(len(channels) -1):
                # CONV, DROPOUT, BATCHNORM, RELU
                main_layers.append(nn.Conv2d(all_channels[i], all_channels[i+1], kernel_sizes[i], stride = 1,padding = kernel_sizes[i]//2))
                if dropout:
                    main_layers.append(nn.Dropout2d(dropout))
                if batchnorm == True:
                    main_layers.append(nn.BatchNorm2d(all_channels[i+1]))
                main_layers.append(non_lin_class(**activation_params))

            main_layers.append(nn.Conv2d(channels[-2], channels[-1], kernel_sizes[-1], stride = 1, padding = kernel_sizes[-1]//2))
        
        self.main_path = nn.Sequential(*main_layers)
        
        sc_layers = []
        if channels[-1] != in_channels:
            sc_layers.append(nn.Conv2d(in_channels, channels[-1], 1, bias = False))
        else:
            sc_layers.append(nn.Identity())               
            
        self.shortcut_path = nn.Sequential(*sc_layers)       


        # ========================

    def forward(self, x: Tensor):
        # TODO: Implement the forward pass. Save the main and residual path to `out`.
        out: Tensor = None
        # ====== YOUR CODE: ======
        out = self.main_path(x) + self.shortcut_path(x)
        # ========================
        out = torch.relu(out)
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. NOT the outer projections)
            The length determines the number of convolutions, EXCLUDING the
            block input and output convolutions.
            For example, if in_out_channels=10 and inner_channels=[5],
            the block will have three convolutions, with channels 10->5->5->10.
            The first and last arrows are the 1X1 projection convolutions, 
            and the middle one is the inner convolution (corresponding to the kernel size listed in "inner kernel sizes").
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        assert len(inner_channels) > 0
        assert len(inner_channels) == len(inner_kernel_sizes)

        # TODO:
        #  Initialize the base class in the right way to produce the bottleneck block
        #  architecture.
        # ====== YOUR CODE: ======
        super().__init__(in_channels = in_out_channels,
                            channels = [inner_channels[0]] + inner_channels + [in_out_channels],
                            kernel_sizes = [1] + inner_kernel_sizes + [1],
                            **kwargs)
        # ========================


class ResNet(CNN):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        bottleneck: bool = False,
        **kwargs,
    ):
        """
        See arguments of CNN & ResidualBlock.
        :param bottleneck: Whether to use a ResidualBottleneckBlock to group together
            pool_every convolutions, instead of a ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions (make sure to use the right stride and padding).
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        #  - Use bottleneck blocks if requested and if the number of input and output
        #    channels match for each group of P convolutions.
        #    Reminder: the number of convolutions performed in the bottleneck block is:
        #    2 + len(inner_channels). [1 for each 1X1 proection convolution] + [# inner convolutions].
        # - Use batchnorm and dropout as requested.
        # ====== YOUR CODE: ======
        N = len(self.channels)
        P = self.pool_every
        
        self.non_lin_class = ACTIVATIONS[self.activation_type]
        pooling_class = POOLINGS[self.pooling_type]
        activation_params = ACTIVATION_DEFAULT_KWARGS[self.non_lin_class]
        
        all_channels = [in_channels] + self.channels
            
        if self.bottleneck == False:
            
            for i in range(N//P):
                # add layers: (CONV -> ACT)*P -> POOL
                layers.append(ResidualBlock(all_channels[i*P], all_channels[i*P + 1: (i+1)*P + 1], [3]*P,
                                            activation_type = self.activation_type, activation_params = activation_params,
                                            batchnorm = self.batchnorm, dropout = self.dropout))
                # add POOL
                layers.append(pooling_class(**self.pooling_params))
                
        else:
            # add first layers: ->(CONV -> ACT)*P -> POOL
            layers.append(ResidualBlock(in_channels, self.channels[0: P], [3]*P,
                                            activation_type = self.activation_type, activation_params = activation_params,
                                            batchnorm = self.batchnorm, dropout = self.dropout))
            layers.append(pooling_class(**self.pooling_params))
            
            for i in range(1,N//P):
                # add layers: (CONV -> ACT)*P -> POOL
                layers.append(ResidualBottleneckBlock(self.channels[i*P], self.channels[i*P + 1: (i+1)*P -1], [3]*(P-2),
                                            activation_type = self.activation_type, activation_params = activation_params,
                                            batchnorm = self.batchnorm, dropout = self.dropout))

                layers.append(pooling_class(**self.pooling_params))

        # left over layer for both cases (bottleneck / not bottleneck)
        if N%P != 0:
                layers.append(ResidualBlock(all_channels[(N//P) * P], all_channels[(N//P) * P + 1: N + 1], [3]*(N%P),
                                            activation_type = self.activation_type, activation_params = activation_params,
                                            batchnorm = self.batchnorm, dropout = self.dropout))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

