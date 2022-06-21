import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import copy

class Linear_QNet(nn.Module):
    def __init__(self, layers_sizes, model_folder_path = './model'):
        super().__init__()
        self.layers_num = len(layers_sizes)
        
        self.model_folder_path = model_folder_path
        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)
        

        # TODO: more than 3 layers
        
        # input_layer_size = layer_sizes[0] 
        # output_layer_size = layer_sizes[-1] 

        # for i in range(self.layers_num-1):
        self.linear1 = nn.Linear(layers_sizes[0], layers_sizes[1])
        self.linear2 = nn.Linear(layers_sizes[1], layers_sizes[2])
        self.linear3 = nn.Linear(layers_sizes[2], layers_sizes[3])

    def forward(self, x):
        # for i in range(len(self.linear)-1):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, file_name='model.pth'):
        file_name = os.path.join(self.model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def restore_from_saved(self, file_name='model.pth'):
        file_name = os.path.join(self.model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))



class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)
        pred_target = self.target_model(next_state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = 0
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(pred_target[idx])
                # Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            else:
                Q_new = reward[idx]

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

        return loss



