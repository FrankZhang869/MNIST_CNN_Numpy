class Max_Pool:    
    def __init__ (self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
    def forward (self, X):
        self.X = X
        B, C, H, W = X.shape
        p = self.pool_size
        s = self.stride
        out_h = (H-p) // s + 1
        out_w = (W-p) // s + 1
        out = np.zeros((B, C, out_h, out_w))
        self.max_idx = {}

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * s
                        h_end = h_start + p
                        w_start = j * s
                        w_end = w_start + p
                        window = X[b, c, h_start:h_end, w_start:w_end]
                        out[b, c, i, j] = np.max(window)
                        self.max_idx[(b, c, i, j)] = np.unravel_index(
                            np.argmax(window), window.shape
                        )
        return out
                        
    def backward (self, d_out):
        B, C, H, W = self.X.shape
        p = self.pool_size
        s = self.stride
        dX = np.zeros_like(self.X)
        for b in range(B):
            for c in range(C):
                for i in range(d_out.shape[2]):
                    for j in range(d_out.shape[3]):
                        h_start = i * s
                        w_start = j * s
                        idx = self.max_idx[(b, c, i, j)]
                        dX[b, c, h_start + idx[0], w_start + idx[1]] += d_out[b, c, i, j]
        return dX

