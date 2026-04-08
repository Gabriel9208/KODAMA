import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, model, layers:list[int]):
        super().__init__()
        self.model = model
        self.layers = layers
        self.features = {}
        self.handles = []

        for layer in self.layers:
            self.handles.append(self.model.model.model[layer].register_forward_hook(self._hook_fn(layer)))

    def __del__(self):
        for handle in self.handles:
            handle.remove()
        self.handles = [] 

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()
            
    def _hook_fn(self, name):
        def hook(module, input, output):
            self.features[name]  = output
        return hook

    def forward(self, X):
        self.features = {}

        self.model(X)
        return self.features
        