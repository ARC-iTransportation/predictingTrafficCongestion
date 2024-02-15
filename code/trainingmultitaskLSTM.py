import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset
from MultitaskLSTM import MultiTaskModel
from custom_loss import RMSELoss
from torch.nn import MSELoss
from torch.nn import L1Loss

class trainMultiTaskLSTM:
    def __init__(self, congestion_path, num_task) -> None:
        self.congestion_path = congestion_path

        self.k = num_task

    def preprocess(self, split_ratio, time_steps, batch_size):
            train_dataloaders = []
            test_dataloaders = []

            congestionlength_df = pd.read_csv(self.congestion_path, sep=',', index_col=0).iloc[:6600, :]
            train, test = train_test_split(congestionlength_df, test_size=split_ratio, shuffle=False)

            train = train.values.T
            test = test.values.T
            n_sample = train.shape[1]-time_steps-1

            for i in range(self.k):
                input_data = np.zeros((n_sample, time_steps, 1))
                correct_data = np.zeros((n_sample, 1))
                test_input_data = np.zeros((test.shape[1]-time_steps-1, time_steps, 1))
                test_correct_data = np.zeros((test.shape[1]-time_steps-1, 1))

                for j in range(n_sample):

                    input_data[j] = train[i, j:j+time_steps].reshape(-1, 1)
                    correct_data[j] = train[i, j+time_steps:j+time_steps+1]
                for j in range(test.shape[1]-time_steps-1):
                    test_input_data[j] = test[i, j:j+time_steps].reshape(-1, 1)
                    test_correct_data[j] = test[i, j+time_steps:j+time_steps+1]
                

                input_data = torch.tensor(input_data, dtype=torch.float)
                correct_data = torch.tensor(correct_data, dtype=torch.float)
                test_input_data = torch.tensor(test_input_data, dtype=torch.float)
                test_correct_data = torch.tensor(test_correct_data, dtype=torch.float)

                train_dataset = TensorDataset(input_data, correct_data)
                train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
                test_dataset = TensorDataset(test_input_data, test_correct_data)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

                train_dataloaders.append(train_loader)
                test_dataloaders.append(test_loader)

            return train_dataloaders, test_dataloaders
    
    def train(self, model, train_loaders, optimizer, criterion, device):
        train_loss = []
        
        model.train()
        model.to(device)

        for data in zip(*train_loaders):
            for j in range(len(data)):
                data[j][0] = data[j][0].to(device)
            optimizer.zero_grad()

            inputs = [data[j][0] for j in range(len(data))]
            outputs = model(inputs)
            for j in range(len(outputs)):
                outputs[j] = outputs[j].to("cpu")   
            losses = [criterion(outputs[j], data[j][1]) for j in range(len(outputs))]

            loss = sum(losses) / 5
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        total_loss = np.mean(train_loss)
        return total_loss
    def evaluate(self, model, test_loader, criterion, device):
        losses_list = []
        model.eval()
        model.to(device)
        with torch.no_grad():
            for data in zip(*test_loader):
                for j in range(len(data)):
                    data[j][0] = data[j][0].to(device)
                inputs = [data[j][0] for j in range(len(data))]

                outputs = model(inputs)
                for j in range(len(outputs)):
                    outputs[j] = outputs[j].to("cpu")
                losses = [criterion(outputs[j], data[j][1]) for j in range(len(outputs))]
                losses_list.append([loss.item() for loss in losses])
        losses = np.mean(losses_list, axis=0)
        return losses
def main():
    congestion_path = "../data/congestion_length_5roads_1hr.csv"
    n_input = 1
    n_output = 1
    n_hidden = 64
    n_layers = 1
    n_task = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(n_input, n_output, n_hidden, n_layers, n_task)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = MSELoss()
    epochs = 100
    batch_size = 35
    time_steps = 24
    split_ratio = 0.2
    obj = trainMultiTaskLSTM(congestion_path, n_task)
    train_loaders, test_loaders = obj.preprocess(split_ratio, batch_size, time_steps)

    for epoch in range(epochs+1):
        total_loss = obj.train(model, train_loaders, optimizer, criterion, device)
        if epoch % epochs == 0:
            print(f"Epoch: {epoch}, Loss: {total_loss}")

    losses = obj.evaluate(model, test_loaders, criterion, device)
    for i, loss in enumerate(losses):
        print(f"Task {i+1} Loss: {loss}")
    # torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    main()