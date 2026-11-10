from builtins import range
import numpy as np

# import numexpr as ne # ~~DELETE LINE~~


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None

    # Get the number of examples
    N = x.shape[0]

    # Flatten each example into a row
    x_flat = x.reshape(N, -1)

    # Compute the affine transformation
    out = x_flat.dot(w) + b

    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    
    # Flatten the input
    N = x.shape[0]
    x_flat = x.reshape(N, -1)

    # Compute gradients
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x_flat.T.dot(dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    
    # Apply ReLU activation element-wise
    out = np.maximum(0, x)

    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    
    # Backpropagate only through positive inputs
    dx = dout * (x > 0)

    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        
        # Compute minibatch statistics
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)

        # Normalize the input
        x_centered = x - sample_mean
        inv_std = 1.0 / np.sqrt(sample_var + eps)
        x_hat = x_centered * inv_std

        # Scale and shift
        out = gamma * x_hat + beta

        # Save values for backward pass
        cache = (x_hat, gamma, x_centered, inv_std)

        # Update running statistics
        running_mean = (
            momentum * running_mean
            + (1 - momentum) * sample_mean
        )
        running_var = (
            momentum * running_var
            + (1 - momentum) * sample_var
        )

    elif mode == "test":
        
        
        # Normalize using running statistics
        x_hat = (x - running_mean) / np.sqrt(
            running_var + eps
        )

        # Scale and shift
        out = gamma * x_hat + beta

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    
    # Unpack cached values
    x_hat, gamma, x_centered, inv_std = cache
    N = dout.shape[0]

    # Gradients for gamma and beta
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)

    # Backpropagate through normalization
    dx_hat = dout * gamma

    dvar = np.sum(
        dx_hat * x_centered * (-0.5) * inv_std**3,
        axis=0
    )

    dmean = (
        np.sum(-dx_hat * inv_std, axis=0)
        + dvar * np.mean(-2 * x_centered, axis=0)
    )

    dx = (
        dx_hat * inv_std
        + (2.0 / N) * x_centered * dvar
        + dmean / N
    )

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    
    x_hat, gamma, x_centered, inv_std = cache
    N = dout.shape[0]

    # Gradients for gamma and beta
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)

    # Simplified input gradient
    dx_hat = dout * gamma

    dx = (inv_std / N) * (
        N * dx_hat
        - np.sum(dx_hat, axis=0)
        - x_hat * np.sum(dx_hat * x_hat, axis=0)
    )

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    
    # Compute mean and variance per example
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)

    # Normalize each example
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mean) * inv_std

    # Scale and shift
    out = gamma * x_hat + beta

    # Save values for backward pass
    cache = (x_hat, gamma, inv_std)

    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    
    # Unpack cached values
    x_hat, gamma, inv_std = cache
    D = dout.shape[1]

    # Gradients for gamma and beta
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)

    # Backpropagate
    dx_hat = dout * gamma

    dx = (inv_std / D) * (
        D * dx_hat
        - np.sum(dx_hat, axis=1, keepdims=True)
        - x_hat * np.sum(
            dx_hat * x_hat,
            axis=1,
            keepdims=True
        )
    )

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.


    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        
        # Create inverted dropout mask
        mask = (np.random.rand(*x.shape) < p) / p

        # Apply dropout
        out = x * mask

    elif mode == "test":
        
        # No dropout during testing
        out = x


    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        
        # Backpropagate through dropout mask
        dx = dout * mask

    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    stride = conv_param["stride"]
    pad = conv_param["pad"]

    # Compute output dimensions
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride

    # Pad the input
    x_pad = np.pad(
        x,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant"
    )

    out = np.zeros(
        (N, F, H_out, W_out),
        dtype=np.result_type(x, w, b)
    )

    # Perform convolution
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):

                    hs = i * stride
                    ws = j * stride

                    window = x_pad[
                        n, :,
                        hs:hs + HH,
                        ws:ws + WW
                    ]

                    out[n, f, i, j] = (
                        np.sum(window * w[f]) + b[f]
                    )

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    
    x, w, b, conv_param = cache

    stride = conv_param["stride"]
    pad = conv_param["pad"]

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    _, _, H_out, W_out = dout.shape

    # Pad input
    x_pad = np.pad(
        x,
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant"
    )

    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.sum(dout, axis=(0, 2, 3))

    # Compute gradients
    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):

                    hs = i * stride
                    ws = j * stride

                    window = x_pad[
                        n, :,
                        hs:hs + HH,
                        ws:ws + WW
                    ]

                    dw[f] += (
                        dout[n, f, i, j] * window
                    )

                    dx_pad[
                        n, :,
                        hs:hs + HH,
                        ws:ws + WW
                    ] += dout[n, f, i, j] * w[f]

    # Remove padding from input gradient
    dx = dx_pad[
        :, :,
        pad:pad + H,
        pad:pad + W
    ]

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    
    N, C, H, W = x.shape

    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    out = np.zeros((N, C, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride

            window = x[
                :,
                :,
                h_start:h_start + pool_height,
                w_start:w_start + pool_width
            ]

            out[:, :, i, j] = np.max(
                window, axis=(2, 3)
            )

    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    
    x, pool_param = cache

    N, C, H, W = x.shape

    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    H_out = dout.shape[2]
    W_out = dout.shape[3]

    dx = np.zeros_like(x)

    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride

            window = x[
                :,
                :,
                h_start:h_start + pool_height,
                w_start:w_start + pool_width
            ]

            max_values = np.max(
                window, axis=(2, 3), keepdims=True
            )

            mask = (window == max_values)

            dx[
                :,
                :,
                h_start:h_start + pool_height,
                w_start:w_start + pool_width
            ] += mask * dout[:, :, i, j][:, :, None, None]

    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    
    N, C, H, W = x.shape

    # Rearrange to (N*H*W, C)
    x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)

    # Apply batch normalization
    out_flat, cache = batchnorm_forward(
        x_flat, gamma, beta, bn_param
    )

    # Restore original shape
    out = out_flat.reshape(
        N, H, W, C
    ).transpose(0, 3, 1, 2)


    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    
    N, C, H, W = dout.shape

    # Rearrange upstream gradients
    dout_flat = dout.transpose(
        0, 2, 3, 1
    ).reshape(-1, C)

    # Compute batch normalization gradients
    dx_flat, dgamma, dbeta = batchnorm_backward(
        dout_flat, cache
    )

    # Restore original shape
    dx = dx_flat.reshape(
        N, H, W, C
    ).transpose(0, 3, 1, 2)


    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    
    N, C, H, W = x.shape

    # Divide channels into G groups
    x_group = x.reshape(N, G, C // G, H, W)

    # Compute mean and variance per group
    mean = np.mean(
        x_group, axis=(2, 3, 4), keepdims=True
    )

    var = np.var(
        x_group, axis=(2, 3, 4), keepdims=True
    )

    # Normalize
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat_group = (x_group - mean) * inv_std

    # Restore original shape
    x_hat = x_hat_group.reshape(N, C, H, W)

    # Scale and shift
    out = gamma * x_hat + beta

    # Save values for backward pass
    cache = (G, x_hat, gamma, inv_std)

    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    
    G, x_hat, gamma, inv_std = cache

    N, C, H, W = dout.shape

    # Gradients for gamma and beta
    dbeta = np.sum(
        dout, axis=(0, 2, 3), keepdims=True
    )

    dgamma = np.sum(
        dout * x_hat, axis=(0, 2, 3), keepdims=True
    )

    # Gradient through scale operation
    dx_hat = dout * gamma

    # Number of elements in each group
    M = (C // G) * H * W

    # Reshape into groups
    dx_hat_group = dx_hat.reshape(N, G, M)
    x_hat_group = x_hat.reshape(N, G, M)

    # Compute input gradient
    dx_group = (
        inv_std.reshape(N, G, 1) / M
    ) * (
        M * dx_hat_group
        - np.sum(dx_hat_group, axis=2, keepdims=True)
        - x_hat_group * np.sum(
            dx_hat_group * x_hat_group,
            axis=2,
            keepdims=True
        )
    )

    # Restore original shape
    dx = dx_group.reshape(N, C, H, W)

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    
    N = x.shape[0]

    # Scores of the correct classes
    correct_scores = x[np.arange(N), y]

    # Compute margins
    margins = np.maximum(
        0,
        x - correct_scores[:, None] + 1
    )

    margins[np.arange(N), y] = 0

    # Compute loss
    loss = np.sum(margins) / N

    # Compute gradient
    mask = (margins > 0).astype(float)

    positive_counts = np.sum(mask, axis=1)
    mask[np.arange(N), y] = -positive_counts

    dx = mask / N

    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    
    N = x.shape[0]

    # Numerical stability
    shifted_scores = x - np.max(
        x, axis=1, keepdims=True
    )

    # Compute Softmax probabilities
    exp_scores = np.exp(shifted_scores)

    probs = exp_scores / np.sum(
        exp_scores, axis=1, keepdims=True
    )

    # Compute loss
    log_sum_exp = np.log(
        np.sum(exp_scores, axis=1)
    )

    correct_scores = shifted_scores[
        np.arange(N), y
    ]

    loss = np.mean(
        -correct_scores + log_sum_exp
    )

    # Compute gradient
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx
