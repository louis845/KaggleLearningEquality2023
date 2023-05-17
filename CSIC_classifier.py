import torch

class Classifier(torch.nn.Module):
    """
    Classifier class. Feedforward nn with 5 total layers and ELU activations. Final layer dim=1 with sigmoid activation. Input dim=1536, and hidden layer dims are 768.
    """
    def __init__(self, input_gaussian_noise=0.0, hidden_dropout_rate=0.1):
        super(Classifier, self).__init__()
        self.input_gaussian_noise = input_gaussian_noise
        self.hidden_dropout_rate = hidden_dropout_rate
        self.fc1 = torch.nn.Linear(1536, 768)
        self.fc2 = torch.nn.Linear(768, 768)
        self.fc3 = torch.nn.Linear(768, 768)
        self.fc4 = torch.nn.Linear(768, 768)
        self.fc5 = torch.nn.Linear(768, 1)
        self.elu = torch.nn.ELU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=self.hidden_dropout_rate)

    def forward(self, x, eval=False):
        if eval:
            x = self.elu(self.fc1(x))
            x = self.elu(self.fc2(x))
            x = self.elu(self.fc3(x))
            x = self.elu(self.fc4(x))
            x = self.sigmoid(self.fc5(x))
        else:
            x = x + torch.randn(x.shape, device=x.device) * self.input_gaussian_noise
            x = self.dropout(self.elu(self.fc1(x)))
            x = self.dropout(self.elu(self.fc2(x)))
            x = self.dropout(self.elu(self.fc3(x)))
            x = self.dropout(self.elu(self.fc4(x)))
            x = self.sigmoid(self.fc5(x))
        return x