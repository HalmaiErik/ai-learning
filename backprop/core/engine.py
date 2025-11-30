import numpy as np

class Tensor:
    def __init__(self, data: np.NDArray[np.float64], children=(), op='', label=''):
        self.data = data
        self.children = set(children)
        self.op = op
        self.label = label

        self.grad = np.zeros(data.shape)
        self._backward = lambda: None

    def __add__(self, other: Tensor):
        out = Tensor(self.data + other.data, (self, other), '+')

        def backward():
            self.grad = self.grad + out.grad
            other.grad = other.grad + out.grad

            if self.grad.shape != self.data.shape:
                list_dims, list_not_keeps = self.get_broadcast_dims(self.data, self.grad)
                for dim in list_dims:
                    if dim in list_not_keeps:
                        self.grad = self.grad.sum(dim, keepdims=False)
                    else:
                        self.grad = self.grad.sum(dim, keepdims=True)

            if other.grad.shape != other.data.shape:
                list_dims, list_not_keeps = self.get_broadcast_dims(other.data, other.grad)
                for dim in list_dims:
                    if dim in list_not_keeps:
                        other.grad = other.grad.sum(dim, keepdims=False)
                    else:
                        other.grad = other.grad.sum(dim, keepdims=True)
        
        self._backward = backward

        return out

    def __matmul__(self, other: Tensor):
        out = Tensor(self.data @ other.data, (self, other), '@')

        def backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        self._backward = backward

        return out

    def tanh(self):
        t = (np.exp(2 * self.data) - 1) / (np.exp(2 * self.data) + 1)
        out = Tensor(t, (self, ), 'tanh')

        def backward():
            self.grad += (1 - t ** 2) * out.grad

        self._backward = backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v in visited:
                return
            
            visited.add(v)
            for child in v.children:
                build_topo(child)
            
            topo.append(v)
        
        build_topo(self)

        self.grad = np.ones(self.data.shape)
        for node in reversed(topo):
            node._backward()

    def get_broadcast_dims(self, input, output):
        list_dims = []
        list_not_keeps = []

        if input.ndim < output.ndim:
            table = np.zeros(input.ndim, output.ndim)
            for i, v_i in enumerate(input.shape):
                for j, v_j in enumerate(output.shape):
                    if v_i == v_j and all(table[i, :j] == 0):  # just accept one-to-one mapping
                        table[i, j] = 1

            for k in range(output.ndim):
                if all(table[:, k] == 0):  # add dimension here
                    np.expand_dims(input, k)
                    list_not_keeps.append(k)

        for i, (l1, l2) in enumerate(zip(input.shape, output.shape)):
            if l1 < l2:
                list_dims.append(i)

        return list_dims, set(list_not_keeps)
                
    
    def __repr__(self):
        return f"Tensor(data={self.data})"

        
