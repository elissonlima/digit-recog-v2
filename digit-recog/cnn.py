import numpy as np
from mlp import MLP

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

class CNN(object):

    def __init__(self,layer_spec, input_size):
        """
        Create CNN Class

        Argument: 
        layer_spec -- python list  of tuples with CNN arch specs 
            ('conv',filter_size, num_filter, pad, stride, activation_function) or
            ('pool', mode, size, stride ) or
            ('fully-conected', out_size, activation_function)
        """
        np.random.seed(1)
        self.layers = []
        for i in range(len(layer_spec)):
            l_spec = {"type":layer_spec[i][0]}

            if i > 0:
                input_size = self.layers[i - 1]["out_size"]

            if l_spec["type"] == 'conv':
                l_spec["W"] = np.random.randn(
                       layer_spec[i][1], #Filter size
                       layer_spec[i][1], #Filter size
                       input_size[-1],
                       layer_spec[i][2]) #Filter Num
                l_spec["b"] = np.random.randn(1,1,1,
                        layer_spec[i][2])
                l_spec["activation"] = np.vectorize(
                        globals()[layer_spec[i][5]])
                l_spec["activation_prime"] = np.vectorize(
                    globals()[layer_spec[i][5]+"_prime"] )
                l_spec["hparameters"] = {
                    "stride" : layer_spec[i][4],
                    "pad" : layer_spec[i][3]
                }
                H_prev = input_size[0] #Prev Out Height
                W_prev = input_size[1] #Prev Out Widht
                n_C = layer_spec[i][2] #Filter Number
                
                n_H = int((H_prev - layer_spec[i][1] + 2 * 
                    layer_spec[i][3]) / layer_spec[i][4]) + 1
                n_W = int((W_prev - layer_spec[i][1] + 2 * 
                    layer_spec[i][3]) / layer_spec[i][4]) + 1

                l_spec["out_size"] = (n_H, n_W, n_C)
                self.layers.append(l_spec)
            elif l_spec["type"] == 'pool':
                l_spec["mode"] = layer_spec[i][1]
                l_spec["hparameters"] = {
                    "f" : layer_spec[i][2], #Filter Size
                    "stride" : layer_spec[i][3]
                }

                H_prev = input_size[0] #Prev Out Height
                W_prev = input_size[1] #Prev Out Widht
                n_C = input_size[2] #Prev Out Depth
                
                n_H = int(1 + (H_prev - layer_spec[i][2]) / \
                        layer_spec[i][3])
                n_W = int(1 + (W_prev - layer_spec[i][2]) / \
                        layer_spec[i][3])

                l_spec["out_size"] = (n_H, n_W, n_C)
                self.layers.append(l_spec)
            elif l_spec["type"] == 'fully-conected':
                l_spec["input_size"] = np.prod(
                    self.layers[i - 1]["out_size"])
                l_spec["out_size"] = layer_spec[i][1]
                l_spec["activation"] = np.vectorize(
                        globals()[layer_spec[i][2]])
                l_spec["activation_prime"] = np.vectorize(
                    globals()[layer_spec[i][2]+"_prime"] )
                l_spec["W"] = np.random.randn(l_spec["out_size"],
                             l_spec["input_size"])
                l_spec["b"] = np.random.randn(l_spec["out_size"], 1)
                self.layers.append(l_spec)



    def feedforward(self, in_):
        for layer in self.layers:
            if layer["type"] == 'conv':
                W = layer["W"]
                b = layer["b"]
                hparam = layer["hparameters"]
                in_ = self.conv_forward(in_,W,b,hparam)[0]
            elif layer["type"] == 'pool':
                in_ = self.pool_forward(in_, 
                    layer["hparameters"], mode=layer["mode"])[0]
            elif layer["type"] == 'fully-conected':

                    if len(in_.shape) > 3: #Mean that prev layer was convolutional
                        in_ = in_.reshape((in_.shape[0], 
                                    np.prod(in_.shape[1:]),1))
                    in_ = self.fully_connected_forward(
                            in_, layer["W"],
                            layer["b"], layer["activation"])
            
            
        print(np.argmax(in_[0]))

    def fully_connected_forward(self, in_, W, b, act_func):
        input_batch_size = in_.shape[0]
        out_ = np.ones((input_batch_size, W.shape[0], 1))
        for m in range(input_batch_size):
            out_[m] = act_func(np.dot(W, in_[m]) + b)

        return out_


    def zero_pad(self,X, pad):
        """
        Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image, 
        as illustrated in Figure 1.
        
        Argument:
        X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
        pad -- integer, amount of padding around each image on vertical and horizontal dimensions
        
        Returns:
        X_pad -- padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
        """
        
        ### START CODE HERE ### (≈ 1 line)
        X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant', constant_values = (0,0))
        ### END CODE HERE ###
        
        return X_pad

    def conv_single_step(self, a_slice_prev, W, b):
        """
        Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
        of the previous layer.
        
        Arguments:
        a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
        W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
        b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
        
        Returns:
        Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
        """

        ### START CODE HERE ### (≈ 2 lines of code)
        # Element-wise product between a_slice_prev and W. Do not add the bias yet.
        s = a_slice_prev * W
        # Sum over all entries of the volume s.
        Z = np.sum(s) + float(b)
        # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
        ### END CODE HERE ###

        return Z

    # GRADED FUNCTION: conv_forward

    def conv_forward(self, A_prev, W, b, hparameters):
        """
        Implements the forward propagation for a convolution function
        
        Arguments:
        A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
        b -- Biases, numpy array of shape (1, 1, 1, n_C)
        hparameters -- python dictionary containing "stride" and "pad"
            
        Returns:
        Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward() function
        """
        
        ### START CODE HERE ###
        # Retrieve dimensions from A_prev's shape (≈1 line)  
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        # Retrieve dimensions from W's shape (≈1 line)
        (f, f, n_C_prev_, n_C) = W.shape

        assert n_C_prev == n_C_prev_
        
        # Retrieve information from "hparameters" (≈2 lines)
        stride = hparameters["stride"]
        pad = hparameters["pad"]
        
        # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
        n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
        n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
        
        # Initialize the output volume Z with zeros. (≈1 line)
        Z = np.zeros((m, n_H, n_W, n_C))
        
        # Create A_prev_pad by padding A_prev
        A_prev_pad = self.zero_pad(A_prev,pad)
        
        for i in range(m):                               # loop over the batch of training examples
            a_prev_pad = A_prev_pad[i]                          # Select ith training example's padded activation
            for h in range(n_H):  
                for w in range(n_W):                       # loop over horizontal axis of the output volume
                    for c in range(n_C):                   # loop over channels (= #filters) of the output volume
                        
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f
                        
                        # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:]
                        #print(a_slice_prev.shape)
                        # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                        Z[i, h, w, c] = self.conv_single_step(a_slice_prev, W[...,c], b[...,c])
                                            
        ### END CODE HERE ###
        
        # Making sure your output shape is correct
        assert(Z.shape == (m, n_H, n_W, n_C))
        
        # Save information in "cache" for the backprop
        cache = (A_prev, W, b, hparameters)
        
        return Z, cache


    def pool_forward(self, A_prev, hparameters, mode = "max"):
        """
        Implements the forward pass of the pooling layer
        
        Arguments:
        A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        hparameters -- python dictionary containing "f" and "stride"
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
        
        Returns:
        A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
        """
        
        # Retrieve dimensions from the input shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        # Retrieve hyperparameters from "hparameters"
        f = hparameters["f"]
        stride = hparameters["stride"]
        
        # Define the dimensions of the output
        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev
        
        # Initialize output matrix A
        A = np.zeros((m, n_H, n_W, n_C))              
        
        ### START CODE HERE ###
        for i in range(m):                         # loop over the training examples
            a_prev = A_prev[i]
            for h in range(n_H):                     # loop on the vertical axis of the output volume
                for w in range(n_W):                 # loop on the horizontal axis of the output volume
                    for c in range (n_C):            # loop over the channels of the output volume
                        
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f
                        
                        # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end,c]
                        
                        # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                        if mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif mode == "average":
                            A[i, h, w, c] = np.mean(a_prev_slice)
        
        ### END CODE HERE ###
        
        # Store the input and hparameters in "cache" for pool_backward()
        cache = (A_prev, hparameters)
        
        # Making sure your output shape is correct
        assert(A.shape == (m, n_H, n_W, n_C))
        
        return A, cache

    def conv_backward(self, dZ, cache):
        """
        Implement the backward propagation for a convolution function
        
        Arguments:
        dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward(), output of conv_forward()
        
        Returns:
        dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
                numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        dW -- gradient of the cost with respect to the weights of the conv layer (W)
            numpy array of shape (f, f, n_C_prev, n_C)
        db -- gradient of the cost with respect to the biases of the conv layer (b)
            numpy array of shape (1, 1, 1, n_C)
        """
        
        ### START CODE HERE ###
        # Retrieve information from "cache"
        (A_prev, W, _, hparameters) = cache
        
        # Retrieve dimensions from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        # Retrieve dimensions from W's shape
        (f, f, n_C_prev, n_C) = W.shape
        
        # Retrieve information from "hparameters"
        stride = hparameters['stride']
        pad = hparameters['pad']
        
        # Retrieve dimensions from dZ's shape
        (m, n_H, n_W, n_C) = dZ.shape
        
        # Initialize dA_prev, dW, db with the correct shapes
        dA_prev = np.zeros(A_prev.shape)
        dW = np.zeros(W.shape)
        db = np.zeros((1,1,1,n_C))

        # Pad A_prev and dA_prev
        A_prev_pad = self.zero_pad(A_prev,pad)
        dA_prev_pad = self.zero_pad(dA_prev,pad)
        
        for i in range(m):                       # loop over the training examples
            
            # select ith training example from A_prev_pad and dA_prev_pad
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            
            for h in range(n_H):                   # loop over vertical axis of the output volume
                for w in range(n_W):               # loop over horizontal axis of the output volume
                    for c in range(n_C):           # loop over the channels of the output volume
                        
                        # Find the corners of the current "slice"
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f
                        
                        # Use the corners to define the slice from a_prev_pad
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:]

                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] +=  W[:,:,:,c] * dZ[i, h, w, c]
                        dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                        db[:,:,:,c] += dZ[i, h, w, c]
                        
            # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        ### END CODE HERE ###
        
        # Making sure your output shape is correct
        assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
        
        return dA_prev, dW, db


    def create_mask_from_window(self, x):
        """
        Creates a mask from an input matrix x, to identify the max entry of x.
        
        Arguments:
        x -- Array of shape (f, f)
        
        Returns:
        mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
        """
        
        ### START CODE HERE ### (≈1 line)
        mask = x == np.max(x)
        ### END CODE HERE ###
        
        return mask

    def distribute_value(self, dz, shape):
        """
        Distributes the input value in the matrix of dimension shape
        
        Arguments:
        dz -- input scalar
        shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
        
        Returns:
        a -- Array of size (n_H, n_W) for which we distributed the value of dz
        """
        
        ### START CODE HERE ###
        # Retrieve dimensions from shape (≈1 line)
        (n_H, n_W) = shape
        
        # Compute the value to distribute on the matrix (≈1 line)
        average = dz / ( n_H * n_W)
        
        # Create a matrix where every entry is the "average" value (≈1 line)
        a = np.ones(shape) * average
        ### END CODE HERE ###
        
        return a

    def pool_backward(self, dA, cache, mode = "max"):
        """
        Implements the backward pass of the pooling layer
        
        Arguments:
        dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
        cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
        
        Returns:
        dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
        """
        
        ### START CODE HERE ###
        
        # Retrieve information from cache (≈1 line)
        (A_prev, hparameters) = cache
        
        # Retrieve hyperparameters from "hparameters" (≈2 lines)
        #stride = hparameters["stride"]
        f = hparameters["f"]
        
        # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
        m, _, _, _ = A_prev.shape
        m, n_H, n_W, n_C = dA.shape
        
        # Initialize dA_prev with zeros (≈1 line)
        dA_prev = np.zeros(A_prev.shape)
        
        for i in range(m):                       # loop over the training examples
            
            # select training example from A_prev (≈1 line)
            a_prev = A_prev[i]
            
            for h in range(n_H):                   # loop on the vertical axis
                for w in range(n_W):               # loop on the horizontal axis
                    for c in range(n_C):           # loop over the channels (depth)
                        
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h
                        vert_end = vert_start + f
                        horiz_start = w
                        horiz_end = horiz_start + f
                        
                        # Compute the backward propagation in both modes.
                        if mode == "max":
                            
                            # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                            a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                            # Create the mask from a_prev_slice (≈1 line)
                            mask = self.create_mask_from_window(a_prev_slice)
                            # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                            dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, dA[i, h, w, c])
                            
                        elif mode == "average":
                            
                            # Get the value a from dA (≈1 line)
                            da = dA[i, h, w, c]
                            # Define the shape of the filter as fxf (≈1 line)
                            shape = (f, f)
                            # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += self.distribute_value(da, shape)
                            
        ### END CODE ###
        
        # Making sure your output shape is correct
        assert(dA_prev.shape == A_prev.shape)
        
        return dA_prev