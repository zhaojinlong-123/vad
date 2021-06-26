from torch import Tensor, nn
import torch

class BoostedDNN(nn.Module):
    def __init__(self,window_feature_size: int,context_window_size: int,first_hidden_features=512,second_hidden_features=512,dropout=0.2,
    ):
        super(BoostedDNN, self).__init__()

        self.dnn = nn.Sequential(
            nn.Dropout(p=dropout),  # not in reference code
            nn.Linear(window_feature_size, first_hidden_features),
            nn.BatchNorm1d(first_hidden_features),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(first_hidden_features, second_hidden_features),
            nn.BatchNorm1d(second_hidden_features),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(second_hidden_features, context_window_size * 4),
        )
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, features: Tensor):
        # features: (batch_size, window_size, feature_size)

        batch_size, window_size, feature_size = features.size()

        x = features.view(batch_size, -1)
        x = self.dnn(x)
        x = x.view(batch_size, window_size, 4)
        x = self.log_softmax(x)

        return x

if __name__ == "__main__":
    model = BoostedDNN(640,10)
    #x = torch.rand(4, 100, 64)
    x = torch.rand(20,10,64)
    o = model(x)
    # (bs, time, output_dim)
    print(o.shape)