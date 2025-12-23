import torch
import torch.nn.functional as F

class MLP:
    def __init__(self, block_size=3, embedding_size=2, hidden_layer_neurons=100, generator: torch.Generator=None):
        self.block_size = block_size
        self.embedding_size = embedding_size

        self.C = torch.randn((27, embedding_size), generator=generator)

        self.W1 = torch.randn((self.block_size * embedding_size, hidden_layer_neurons), generator=generator)
        self.b1 = torch.randn(hidden_layer_neurons)

        self.W2 = torch.randn((hidden_layer_neurons, 27), generator=generator)
        self.b2 = torch.randn(27)

        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]
        for p in self.parameters:
            p.requires_grad = True

    def generate(self, count: int, itos: dict[int, str], generator: torch.Generator=None):
        for i in range(count):
            word = []
            context = [0] * self.block_size

            while True:
                emb = self.C[torch.tensor([context])] # (1, block_size, embedding_size)
                h = torch.tanh(emb.view(1, -1) @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2
                probs = F.softmax(logits, dim=1)

                ix = torch.multinomial(probs, num_samples=1, generator=generator).item()
                if ix == 0:
                    break

                context = context[1:] + [ix]
                word.append(itos[ix])
            
            print(''.join(word))

    def train(self, X: torch.Tensor, Y: torch.Tensor, epochs: int, learning_rate=0.1, print_losses=True):        
        for epoch in range(epochs):
            # minibatch - train on a batch of X - batch of 32 examples
            # the quality of our gradient is lower, so the direction is not as reliable, but it is good enough
            # better to have an approx gradient and make more steps, than having the exact gradient and doing less steps
            ix = torch.randint(0, X.shape[0], (32, ))

            loss = self.forward(X[ix], Y[ix])
            if print_losses:
                print(epoch + 1, loss.item())

            self.backward(loss, learning_rate, epoch)
        
        return loss

    def forward(self, X: torch.Tensor, Y: torch.Tensor):
         # embedding layer
        emb = self.C[X] # this will be X.shape x 2, which is len(combinations) x 3 x 2. Need to view it as len(combinations) x 6

        # hidden layer
        h = torch.tanh(emb.view(-1, self.block_size * self.embedding_size) @ self.W1 + self.b1)

        # final layer
        logits = h @ self.W2 + self.b2
        '''
        counts = logits.exp()
        prob = counts / counts.sum(1, keepdim=True)

        return prob

        def loss(self, Y_predicted: torch.Tensor, Y: torch.Tensor):
            return -Y_predicted[torch.arange(32), Y].log().mean()
        '''

        '''
         same as commented out code, just more efficient & safe:
            - safe from overflow
            - doesn't create new temporary tensors in memory
            - calculates forward pass in more simpler mathematical operations => more efficient
            - because of the prev point, the backward pass will be simpler and more efficient too
        '''
        loss = F.cross_entropy(logits, Y)
        
        return loss

    def backward(self, loss: torch.Tensor, learning_rate: int, epoch: int):
        for p in self.parameters:
            p.grad = None
        
        loss.backward()

        # decrease learning rate on large epochs
        learning_rate = learning_rate if epoch < 100_000 else learning_rate / 10

        for p in self.parameters:
            p.data += -learning_rate * p.grad
    
