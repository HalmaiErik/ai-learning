import torch
import torch.nn.functional as F

from shared.utils import build_char_index_maps


class SingleLayerBigramNN:
    def __init__(self, generator: torch.Generator):
        self.W = torch.randn((27, 27), generator=generator, requires_grad=True)

    def train(self, iterations: int, training_words: list[str]):
        self.stoi, self.itos = build_char_index_maps(training_words)

        xs, ys = self.build_training_set(training_words)

        for i in range(iterations):
            ys_pred = self.forward(xs)

            loss = self.loss(ys_pred, ys)
            print(i, loss.item())

            self.backward(loss)

            self.W.data += -50 * self.W.grad

    def generate(self, count: int, generator: torch.Generator=None):
        for i in range(count):
            word = []
            word_index = 0

            while True:
                p = self.forward(torch.tensor([word_index]))
                word_index = torch.multinomial(p, num_samples=1, replacement=True, generator=generator).item()

                if word_index == 0:
                    break
                
                word.append(self.itos[word_index])
            
            print(i, ''.join(word))

    def build_training_set(self, words: list[str]):
        xs, ys = [], []
        for w in words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = self.stoi[ch1]
                ix2 = self.stoi[ch2]

                xs.append(ix1)
                ys.append(ix2)
        
        xs = torch.tensor(xs)
        ys = torch.tensor(ys)

        return xs, ys

        
    def forward(self, xs: torch.Tensor):
        # one hot encode input integers
        # xenc.shape = (len(words), 27)
        xenc = F.one_hot(xs, num_classes = 27).float()

        logits = xenc @ self.W # log-counts

        # these 2 lines are called softmax
        counts = logits.exp() # equivalent to bigram.bigram_counts
        probs = counts / counts.sum(1, keepdim=True)

        return probs

    def loss(self, ys_pred: torch.Tensor, ys: torch.Tensor):
        # the last part is regularization, to avoid overfitting
        # it tries to make W values go closer to 0, to achieve more uniform probabilities
        return -ys_pred[torch.arange(len(ys_pred)), ys].log().mean() + 0.01 * (self.W ** 2).mean()

    def backward(self, loss: torch.Tensor):
        self.zero_grad()
        loss.backward()
    
    def zero_grad(self):
        self.W.grad = None

