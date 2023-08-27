import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, load_existing=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model_folder_path = './model'
        self.file_path = os.path.join(self.model_folder_path, 'model.pth')

        if load_existing and os.path.exists(self.file_path):
            print("Loading existing model...")
            self.load_existing_model()
        else:
            print("Creating a new model...")
            self.create_new_model()

    def create_new_model(self):
        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)

    def load_existing_model(self):
        model_state_dict = torch.load(self.file_path)
        self.create_new_model()  # Create the model architecture
        self.load_state_dict(model_state_dict)
        self.eval()  # Set the model to evaluation mode

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, filename='model.pth'):
        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)

        file_path = os.path.join(self.model_folder_path, filename)
        torch.save(self.state_dict(), file_path)



class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):

        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)

        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state,  0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predict Q values with current state
        pred = self.model(state)

        target = pred.clone()

        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done

        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
