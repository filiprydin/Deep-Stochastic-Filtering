import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_size, parameters, verbose = False, normalisation_factor = 0):
        super(SimpleNN, self).__init__()

        # Hyperparameters
        self.hidden_size = parameters["Hidden size"]
        self.learning_rate = parameters["Learning rate"]
        self.epochs = parameters["Max epochs"]
        self.batch_size = parameters["Batch size"]
        self.patience = parameters["Early stopping patience"]
        self.verbose = verbose

        self.norm_targets = parameters["Normalise targets"]
        self.norm_const_targets = parameters["Normalisation constant targets"]
        
        self.tail_terms = parameters["Tail terms"]
        self.sde_state_dim = parameters["State dim"]
        
        self.only_pos = parameters["Only positive"]

        self.grad_clip = parameters["Gradient clipping"]
        self.grad_clip_value = parameters["Gradient clip value"]

        if not self.tail_terms:
            self.output_size = 1
        else:
            self.output_size = self.sde_state_dim + 3

        # Normalisation factor 
        self.norm_factor = normalisation_factor

        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        x = self.fc1(input)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)

        if self.tail_terms:
            x_input = input[:,0:self.sde_state_dim]
            xi1 = x[:,0]
            xi2 = x[:,1]
            xi3 = x[:,2:(2+self.sde_state_dim)]
            xi4 = x[:,2+self.sde_state_dim]
            norm2 = torch.pow(torch.norm(x_input - xi3, p=2, dim=1), 2)
            energy = xi1 + xi2 * norm2 * (norm2 > xi4)
        else:
            energy = x
            
        if not self.only_pos: 
            output = torch.exp(-energy).view(-1,1)
        else: 
            x_input = input[:,0]
            output = torch.exp(-energy).view(-1,1) * (x_input >= 0).view(-1,1)

        if not self.training:
            return output * self.norm_factor
        else: 
            return output
    
    def train_model(self, input_train, target_train, input_val, target_val):
        criterion = nn.MSELoss()

        if self.norm_targets:
            avg_target = torch.mean(target_train)
            self.norm_factor = avg_target / self.norm_const_targets
        else:
            self.norm_factor = torch.tensor(1, dtype = torch.float32)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        early_stopper = EarlyStopper(patience=self.patience, min_delta=0)

        train_dataset = torch.utils.data.TensorDataset(input_train, target_train / self.norm_factor)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for inputs, targets in train_loader:
                outputs = self(inputs)
                train_loss = criterion(outputs, targets)
                optimizer.zero_grad()
                train_loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_value)
                optimizer.step()

            self.train(False)
            outputs_val = self(input_val)
            validation_loss = criterion(outputs_val, target_val)

            if self.verbose:
                if (epoch + 1) % 100 == 0:
                    outputs_train = self(input_train)
                    train_loss = criterion(outputs_train, target_train)
                    print(f"Epoch [{epoch+1}/{self.epochs}]. Training loss: {train_loss.item()}. Validation loss: {validation_loss.item()}", flush=True)
            
            self.train(True)

            if early_stopper.early_stop(validation_loss):   
                if self.verbose:
                    print(f"Early stopped at epoch {epoch} with validation loss {validation_loss}", flush=True)
                break     

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
