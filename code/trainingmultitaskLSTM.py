import pandas as pd
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import MSELoss
from torch.nn import L1Loss

from MultitaskLSTM import MultiTaskModel


class trainMultiTaskLSTM:
    """
    This class is designed for training a multi-task LSTM model.
        Attributes:
            congestion_path: str: The path of the congestion data.
            k: int: The number of tasks.
    """
    def __init__(self, congestion_path: str, num_task: int) -> None:
        self.congestion_path = congestion_path
        self.k = num_task

    def preprocess(self, split_ratio: float, time_steps: int, batch_size: int) -> tuple:
            """
            Prepares the dataset for training by splitting it into train and test sets,
                reshaping, and loading into DataLoader objects for each task. 
                Args:
                    split_ratio: float: The ratio of the test set to the train set.
                    time_steps: int: The number of time steps to consider for each sample.
                    batch_size: int: The number of samples in each batch.
                Returns:
                    tuple: containing lists of DataLoader objects for the train and test sets.
            """
            train_dataloaders = []
            test_dataloaders = []

            congestionlength_df = pd.read_csv(self.congestion_path, sep=',', index_col=0).iloc[:6600, :]
            train, test = train_test_split(congestionlength_df, test_size=split_ratio, shuffle=False) # split the data into train and test set from dataframe

            train = train.values.T # transpose the train data
            test = test.values.T # transpose the test data
            n_sample = train.shape[1]-time_steps-1 # the number of samples in the train data

            ## Create the input and label data for each task
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
                
                ## Convert to tensor
                input_data = torch.tensor(input_data, dtype=torch.float) # train_data: (n_sample, time_steps, 1)
                correct_data = torch.tensor(correct_data, dtype=torch.float) # train_label: (n_sample, 1)
                test_input_data = torch.tensor(test_input_data, dtype=torch.float) # test_data: (n_sample, time_steps, 1)
                test_correct_data = torch.tensor(test_correct_data, dtype=torch.float) # test_label: (n_sample, 1)

                ## Create DataLoader
                train_dataset = TensorDataset(input_data, correct_data) 
                train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
                test_dataset = TensorDataset(test_input_data, test_correct_data)
                test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

                ## Append the DataLoader of each task to the list
                train_dataloaders.append(train_loader)
                test_dataloaders.append(test_loader)

            return train_dataloaders, test_dataloaders
    
    def train(self, model, train_loaders, optimizer, criterion, device) -> float:
        """
        Trains the model on the dataset.
            Args:
                model (torch.nn.Module): The LSTM model to be trained.
                train_loaders (list[DataLoader]): A list of DataLoader objects for training.
                optimizer (torch.optim.Optimizer): The optimizer for training the model.
                criterion: The loss function.
                device: The device (CPU or GPU) to perform the training on.
            Returns:
                The average training loss as a float.
        """
        train_loss = []
        
        model.train()
        model.to(device)

        for data in zip(*train_loaders): # Get the data of each task
            for j in range(len(data)): # Send the input data to the device
                data[j][0] = data[j][0].to(device) # dim1 index: task number, dim2 index 0: input, index 1: label
            optimizer.zero_grad() 

            inputs = [data[j][0] for j in range(len(data))] # Get the input data of each task
            outputs = model(inputs) # predict the output of each task
            for j in range(len(outputs)): # Send the output to the cpu
                outputs[j] = outputs[j].to("cpu") # dim1 index: task number
            losses = [criterion(outputs[j], data[j][1]) for j in range(len(outputs))] # Calculate the loss of each task

            loss = sum(losses) / 5 # Calculate the mean loss of each task
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item()) # Append the loss of each task to the list
        total_loss = np.mean(train_loss) # Calculate the mean loss of each task
        return total_loss
    def evaluate(self, model, test_loader, criterion, device) -> list:
        """
        Evaluates the trained model on the test dataset.
            Args:
                model (torch.nn.Module): The trained LSTM model.
                test_loader (list[DataLoader]): A list of DataLoader objects for testing.
                criterion: The loss function.
                device: The device (CPU or GPU) for evaluation.
            Returns:
                A list of losses for each task.
        """
        losses_list = []
        model.eval()
        model.to(device) # Send the model to the device
        with torch.no_grad():
            for data in zip(*test_loader): # Get the data of each task
                for j in range(len(data)): # Send the data to the device
                    data[j][0] = data[j][0].to(device) # dim1 index: task number, dim2 index 0: input, index 1: label
                inputs = [data[j][0] for j in range(len(data))] # Get the input data of each task

                outputs = model(inputs) # predict the output of each task
                for j in range(len(outputs)): # Send the output to the cpu
                    outputs[j] = outputs[j].to("cpu") # dim1 index: task number
                losses = [criterion(outputs[j], data[j][1]) for j in range(len(outputs))] # Calculate the loss of each task
                losses_list.append([loss.item() for loss in losses]) # Append the loss of each task to the list
        losses = np.mean(losses_list, axis=0) # Calculate the mean loss of each task
        return losses
    
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    congestion_path = cfg.data.input_file
    n_input = cfg.model.n_input
    n_output = cfg.model.n_output
    n_hidden = cfg.model.n_hidden
    n_layers = cfg.model.n_layers
    n_task = cfg.model.n_task
    epochs = cfg.epochs
    batch_size = cfg.batch_size
    time_steps = cfg.time_steps
    split_ratio = cfg.split_ratio
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel(n_input, n_output, n_hidden, n_layers, n_task)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = MSELoss()
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