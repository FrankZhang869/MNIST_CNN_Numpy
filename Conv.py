class Conv:
    def __init__ (self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.W = np.random.randn(
            out_channels,
            in_channels,
            kernel_size,
            kernel_size
        ) * np.sqrt(2 / (in_channels * kernel_size * kernel_size)) # in this case it would be
        # shape (8, 1, 3, 3)
        self.b = np.zeros(out_channels) # shape (8, )
    
    def forward(self, X):
        self.X = X # shape (batch size, in_channels, imageHeight, imageWidth)
        
        # batch size is how many images are processed per epoch
        batch_size, _, H, W = X.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding

        # Output spatial dimensions
        H_out = (H + 2 * P - K) // S + 1
        W_out = (W + 2 * P - K) // S + 1

        # Apply padding
        X_padded = np.pad(
            X,
            ((0, 0), (0, 0), (P, P), (P, P)),
            mode="constant"
        )

        # Output tensor
        out = np.zeros((batch_size, self.out_channels, H_out, W_out))

        # Convolution operation
        for n in range(batch_size): # for each image in the batch
            for f in range(self.out_channels): # for filters in model
                for i in range(H_out): # row of the output matrix
                    for j in range(W_out): # column of the output matrix
                        # these lines basically just define where the window is positioned on the image
                        h_start = i * S
                        h_end = h_start + K
                        w_start = j * S
                        w_end = w_start + K

                        window = X_padded[n, :, h_start:h_end, w_start:w_end] # shape is (in_channels, K, K) - the n paramter is which image
                        out[n, f, i, j] = np.sum(window * self.W[f]) + self.b[f] # this is like the Z term 

        return out
    def backward(self, d_out):
        """
        d_out shape:
        (batch_size, out_channels, H_out, W_out)
        """

        X = self.X
        batch_size, _, H, W = X.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding

        # Initialize gradients
        dX = np.zeros_like(X)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        # Pad X and dX
        X_padded = np.pad(
            X,
            ((0, 0), (0, 0), (P, P), (P, P)),
            mode="constant"
        )
        dX_padded = np.zeros_like(X_padded)

        _, _, H_out, W_out = d_out.shape

        for n in range(batch_size):
            for f in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * S
                        h_end = h_start + K
                        w_start = j * S
                        w_end = w_start + K

                        window = X_padded[n, :, h_start:h_end, w_start:w_end]

                        # Gradients - this is all very similar to backprop in simple neural net
                        dW[f] += d_out[n, f, i, j] * window # derivative of loss w.r.t output cross correlated with the input
                        db[f] += d_out[n, f, i, j] # derivative of loss w.r.t output 
                        dX_padded[n, :, h_start:h_end, w_start:w_end] += (
                            d_out[n, f, i, j] * self.W[f]
                        )

        # Remove padding
        if P > 0:
            dX = dX_padded[:, :, P:-P, P:-P]
        else:
            dX = dX_padded
        self.dW = dW
        self.db = db
        return dX

    def update (self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

