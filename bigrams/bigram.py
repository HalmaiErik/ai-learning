import torch

from utils import build_char_index_maps


class Bigram:
    def __init__(self, words: list[str]):
        self.words = words
        self.stoi, self.itos = build_char_index_maps(self.words)

        self.build_bigram_counts()
        self.build_probability_distributions()

    def build_bigram_counts(self):
        self.bigram_counts = torch.zeros((27, 27), dtype=torch.int32)
        for w in self.words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                self.bigram_counts[self.stoi[ch1], self.stoi[ch2]] += 1

    def build_probability_distributions(self):
        # model smoothing - so that no count is 0, because that would give inf loss for unlikely names
        self.probabilities = (self.bigram_counts + 1).float()
        self.probabilities /= self.probabilities.sum(1, keepdim=True)

    def generate(self, count: int, generator: torch.Generator=None):
        for i in range(count):
            word = []
            word_index = 0

            while True:
                p = self.probabilities[word_index]
                word_index = torch.multinomial(p, num_samples=1, replacement=True, generator=generator).item()

                if word_index == 0:
                    break
                
                word.append(self.itos[word_index])
            
            print(i, ''.join(word))

    '''
    GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
    equivalent to maximizing the log likelihood (because log is monotonic)
    equivalent to minimizing the negative log likelihood
    equivalent to minimizing the average negative log likelihood
    
    log(a*b*c) = log(a) + log(b) + log(c) - so we need to add up the log probabilities
    ''' 
    def loss(self):
        log_likelihood = 0.0
        n = 0
        for w in self.words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                i1 = self.stoi[ch1]
                i2 = self.stoi[ch2]

                prob = self.probabilities[i1, i2]
                log_prob = torch.log(prob)
                log_likelihood += log_prob
                n += 1
        
        # we want a loss function where small number = good
        negative_log_likelihood = -log_likelihood

        # average out the loss functions
        return negative_log_likelihood / n


                