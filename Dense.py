class Dense:
    def __init__ (self, input_dim, output_dim):
        # He initialization of weights scales them to make sure activations don't explode
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.b = np.zeros(output_dim)
    def forward(self, X):
        # X shape is (batch_size, input_dim)
        self.X = X
        return X.dot(self.W) + self.b # first run this yields (batch_size, 128) and second run this yields (batch_size, 10)
    def backward(self, d_out):
        # d_out shape is (batch_size, output_dim)
        self.dW = self.X.T.dot(d_out)
        self.db = np.sum(d_out, axis=0) 
        dX = d_out.dot(self.W.T) 
        return dX
    def update (self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db
