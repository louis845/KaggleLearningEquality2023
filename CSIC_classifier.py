import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Classifier(torch.nn.Module):
    """
    Classifier class. Feedforward nn with 5 total layers and ELU activations. Final layer dim=1 with sigmoid activation. Input dim=1536, and hidden layer dims are 768.
    """
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(1536, 768)
        self.fc2 = torch.nn.Linear(768, 768)
        self.fc3 = torch.nn.Linear(768, 768)
        self.fc4 = torch.nn.Linear(768, 768)
        self.fc5 = torch.nn.Linear(768, 1)
        self.elu = torch.nn.ELU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.elu(self.fc3(x))
        x = self.elu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x