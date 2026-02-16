class CNN:
    def __init__(self, layers):
        self.layers = layers
    def forward (self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    def backward (self, d_out):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        return d_out
    def step (self, lr):
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(lr)
