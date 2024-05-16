
class Model:
    def __init__(self, name, T1map, T2map, T2starmap, PDmap, X_dim, Y_dim, Z_dim, X_dim_res, Y_dim_res, Z_dim_res):
        self.name = name
        self.T1map = T1map
        self.T2map = T2map
        self.T2smap = T2starmap
        self.PDmap = PDmap 
        self.X_dim = X_dim
        self.Y_dim = Y_dim
        self.Z_dim = Z_dim
        self.X_dim_res = X_dim_res
        self.Y_dim_res = Y_dim_res
        self.Z_dim_res = Z_dim_res
        
