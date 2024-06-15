import torch
import torch.nn as nn
import torch.optim as optim

class HybridModel(nn.Module):
    def __init__(self, input_seq_size, input_non_seq_size, parameters, verbose = False, normalisation_factor = 0):
        super(HybridModel, self).__init__()

        # Hyperparameters
        self.hidden_size = parameters["Hidden size"]
        self.hidden_size_dnn = parameters["Hidden size DNN"]
        self.num_layers = parameters["LSTM layers"]
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
        
        # Sequential data processing
        self.rnn = nn.LSTM(input_size=input_seq_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        
        # Non-sequential data processing
        self.non_seq_fc = nn.Linear(input_non_seq_size, self.hidden_size)
        
        # Concatenation layer
        self.concat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size_dnn)  # Concatenating hidden states
        
        # Output layer
        self.output_layer = nn.Linear(self.hidden_size_dnn, self.output_size)  # Regression output
        
    def forward(self, seq_data, non_seq_data):
        # Sequential data processing
        _, (seq_hidden, _) = self.rnn(seq_data)
        seq_hidden = seq_hidden[-1]
        
        # Non-sequential data processing
        non_seq_hidden = torch.relu(self.non_seq_fc(non_seq_data))
        
        # Concatenation
        combined_hidden = torch.cat((seq_hidden, non_seq_hidden), dim=1)
        concatenated_hidden = torch.relu(self.concat_layer(combined_hidden))
        
        # Output layer
        x = self.output_layer(concatenated_hidden)

        if self.tail_terms:
            xi1 = x[:,0]
            xi2 = x[:,1]
            xi3 = x[:,2:(2+self.sde_state_dim)]
            xi4 = x[:,2+self.sde_state_dim]
            norm2 = torch.pow(torch.norm(non_seq_data - xi3, p=2, dim=1), 2)
            energy = xi1 + xi2 * norm2 * (norm2 > xi4)
        else:
            energy = x

        if not self.only_pos:
            output = torch.exp(-energy).view(-1,1)
        else:
            output = torch.exp(-energy).view(-1,1) * (non_seq_data >= 0).view(-1,1)

        if not self.training:
            return output * self.norm_factor
        else: 
            return output

    def train_model(self, input_seq_train, input_non_seq_train, target_train, input_seq_val, input_non_seq_val, target_val):
        criterion = nn.MSELoss()

        if self.norm_targets:
            avg_target = torch.mean(target_train)
            self.norm_factor = avg_target / self.norm_const_targets
        else:
            self.norm_factor = 1

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        early_stopper = EarlyStopper(patience=self.patience, min_delta=0)

        train_dataset = torch.utils.data.TensorDataset(input_seq_train, input_non_seq_train, target_train / self.norm_factor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):

            for seq_data, non_seq_data, target in train_loader:
                optimizer.zero_grad()
                output = self(seq_data, non_seq_data)
                train_loss = criterion(output, target)
                train_loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_value)
                optimizer.step()

            self.train(False)
    
            # Compute validation loss
            batch_size = 100000  # Must run in batches to save memory
            total_validation_loss = 0.0
            total_batches = 0
            for i in range(0, input_seq_val.shape[0], batch_size):
                batch_end = min(i + batch_size, len(input_seq_val))

                input_seq_batch = input_seq_val[i:batch_end,:,:]
                input_non_seq_batch = input_non_seq_val[i:batch_end,:]
                target_batch = target_val[i:batch_end]
                
                outputs_val = self(input_seq_batch, input_non_seq_batch)
                validation_loss = criterion(outputs_val, target_batch)
                total_validation_loss += validation_loss.item()
                total_batches += 1
            validation_loss = total_validation_loss / total_batches

            if self.verbose:
                if (epoch + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}]. Validation loss: {validation_loss}", flush=True)
            
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
