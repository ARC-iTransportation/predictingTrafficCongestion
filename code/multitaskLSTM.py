import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, k):
        super(MultiTaskModel, self).__init__()
        self.feature_extractor = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.task_predictor = nn.ModuleList([nn.Linear(hidden_size, output_size) for _ in range(k)])

    def forward(self, inputs):
        outputs = []
        for i, x in enumerate(inputs):
            features, _ = self.feature_extractor(x)
            task_output = self.task_predictor[i](features[:, -1, :])
            outputs.append(task_output)
        return outputs
