from builtins import range
from builtins import object
import os
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        
        # Initialize first layer weights and biases
        self.params['W1'] = weight_scale * np.random.randn(
            input_dim, hidden_dim
        )
        self.params['b1'] = np.zeros(hidden_dim)

        # Initialize second layer weights and biases
        self.params['W2'] = weight_scale * np.random.randn(
            hidden_dim, num_classes
        )
        self.params['b2'] = np.zeros(num_classes)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        
        # Get weights and biases
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        # First layer: Affine + ReLU
        hidden, cache1 = affine_relu_forward(X, W1, b1)

        # Second layer: Affine
        scores, cache2 = affine_forward(hidden, W2, b2)


        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        
        # Compute Softmax loss
        loss, dscores = softmax_loss(scores, y)

        # Add L2 regularization
        loss += 0.5 * self.reg * (
            np.sum(W1 ** 2) + np.sum(W2 ** 2)
        )

        # Backward pass through second layer
        dhidden, dW2, db2 = affine_backward(
            dscores, cache2
        )

        # Backward pass through first layer
        dx, dW1, db1 = affine_relu_backward(
            dhidden, cache1
        )

        # Store gradients with regularization
        grads['W1'] = dW1 + self.reg * W1
        grads['b1'] = db1

        grads['W2'] = dW2 + self.reg * W2
        grads['b2'] = db2


        return loss, grads

    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True



class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        
        # Define dimensions of all layers
        dims = [input_dim] + list(hidden_dims) + [num_classes]

        # Initialize weights and biases
        for i in range(1, self.num_layers + 1):
            self.params[f'W{i}'] = (
                weight_scale * np.random.randn(
                    dims[i - 1], dims[i]
                )
            )
            self.params[f'b{i}'] = np.zeros(dims[i])

            # Initialize normalization parameters
            if i < self.num_layers and self.normalization in (
                'batchnorm', 'layernorm'
            ):
                self.params[f'gamma{i}'] = np.ones(dims[i])
                self.params[f'beta{i}'] = np.zeros(dims[i])


        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        
        # Initialize forward pass
        hidden = X
        caches = []

        # Forward pass through hidden layers
        for i in range(1, self.num_layers):

            W = self.params[f'W{i}']
            b = self.params[f'b{i}']

            # Affine forward
            hidden, fc_cache = affine_forward(hidden, W, b)

            # Optional normalization
            norm_cache = None

            if self.normalization == 'batchnorm':
                gamma = self.params[f'gamma{i}']
                beta = self.params[f'beta{i}']

                hidden, norm_cache = batchnorm_forward(
                    hidden, gamma, beta, self.bn_params[i - 1]
                )

            elif self.normalization == 'layernorm':
                gamma = self.params[f'gamma{i}']
                beta = self.params[f'beta{i}']

                hidden, norm_cache = layernorm_forward(
                    hidden, gamma, beta, self.bn_params[i - 1]
                )

            # ReLU activation
            hidden, relu_cache = relu_forward(hidden)

            # Optional dropout
            dropout_cache = None

            if self.use_dropout:
                hidden, dropout_cache = dropout_forward(
                    hidden, self.dropout_param
                )

            # Save caches for backward pass
            caches.append(
                (fc_cache, norm_cache, relu_cache, dropout_cache)
            )

        # Final affine layer
        L = self.num_layers

        scores, final_cache = affine_forward(
            hidden,
            self.params[f'W{L}'],
            self.params[f'b{L}']
        )

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        
        # Compute Softmax loss
        loss, dout = softmax_loss(scores, y)

        L = self.num_layers

        # Add L2 regularization
        for i in range(1, L + 1):
            W = self.params[f'W{i}']
            loss += 0.5 * self.reg * np.sum(W ** 2)

        # Backward through final affine layer
        dout, dW, db = affine_backward(
            dout, final_cache
        )

        grads[f'W{L}'] = (
            dW + self.reg * self.params[f'W{L}']
        )
        grads[f'b{L}'] = db

        # Backward through hidden layers
        for i in range(L - 1, 0, -1):

            fc_cache, norm_cache, relu_cache, dropout_cache = (
                caches[i - 1]
            )

            # Dropout backward
            if self.use_dropout:
                dout = dropout_backward(
                    dout, dropout_cache
                )

            # ReLU backward
            dout = relu_backward(dout, relu_cache)

            # Normalization backward
            if self.normalization == 'batchnorm':
                dout, dgamma, dbeta = batchnorm_backward(
                    dout, norm_cache
                )

                grads[f'gamma{i}'] = dgamma
                grads[f'beta{i}'] = dbeta

            elif self.normalization == 'layernorm':
                dout, dgamma, dbeta = layernorm_backward(
                    dout, norm_cache
                )

                grads[f'gamma{i}'] = dgamma
                grads[f'beta{i}'] = dbeta

            # Affine backward
            dout, dW, db = affine_backward(
                dout, fc_cache
            )

            grads[f'W{i}'] = (
                dW + self.reg * self.params[f'W{i}']
            )
            grads[f'b{i}'] = db

        return loss, grads


    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True