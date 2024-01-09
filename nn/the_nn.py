from getdata import returndata
from torch.utils.data import TensorDataset
from sklearn import metrics
from torch.nn import init
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def test_model(predict, y_test):
    predict = predict.detach().numpy()
    y_test = y_test.detach().numpy()
    MAE = metrics.mean_absolute_error(y_test, predict)
    MSE = metrics.mean_squared_error(y_test, predict)
    RMSE = MSE ** 0.5

    return MAE, RMSE

class Network(nn.Module):
    def __init__(self, feat_num):
        super(Network, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(feat_num, 40), nn.ReLU(),
                                     nn.Linear(40, 10), nn.ReLU(),
                                     nn.Linear(10, 1)])
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, val=0)

    def forward(self, x):
        predict = x
        for i, layer in enumerate(self.layers):
            predict = layer(predict)
        return predict

batch_size = 16
learning_rate = 0.001
num_epochs = 2000 

use_gpu = torch.cuda.is_available()
# ['ML_HEA_100_bridge', 'ML_HEA_110_bridge', 'ML_HEA_111_bridge', 'ML_HEA_211_edge', 'ML_HEA_211_hill', 'ML_HEA_211_summit', 'ML_HEA_211_valley', 'ML_HEA_532_higher_edge', 'ML_HEA_532_hill1', 'ML_HEA_532_hill2', 'ML_HEA_532_inner_higher_edge', 'ML_HEA_532_outer_higher_edge', 'ML_HEA_532_valley2']
x_train, y_train, x_valid, y_valid = returndata('all')
y_train = np.array([[y] for y in y_train])
y_valid = np.array([[y] for y in y_valid])
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
x_valid = torch.Tensor(x_valid)
y_valid = torch.Tensor(y_valid)

train_set = TensorDataset(x_train, y_train)
valid_set = TensorDataset(x_valid, y_valid)

train_loader = DataLoader(train_set,
                          batch_size=16,
                          shuffle=True)
valid_loader = DataLoader(valid_set,
                          batch_size=16,
                          shuffle=True)

model = Network(37)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.MSELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

draw_train_MAE = []
draw_train_RMSE = []
draw_train_MSE = []
draw_test_MAE = []
draw_test_RMSE = []
draw_test_MSE = []

t = range(num_epochs)
for epoch in range(num_epochs):
    print('*' * 10)
    print(f'epoch {epoch + 1}')
    running_loss = 0.0
    running_acc = 0.0
    model.train()
    train_MAE = []
    train_RMSE = []
    n_train = 0
    for x, y in train_loader:
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        x = x.to(device)
        y = y.to(device)
        model = model.to(device)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        MAE, RMSE = test_model(out, y)
        train_MAE.append(MAE)
        train_RMSE.append(RMSE)
        n_train += 1
    loss1 = running_loss/n_train
    fin_train_MAE = np.sum(train_MAE) / n_train
    fin_train_RMSE = np.sum(train_RMSE)/ n_train
    draw_train_MAE.append(fin_train_MAE)
    draw_train_RMSE.append(fin_train_RMSE)
    draw_train_MSE.append(loss1)
    print("train_MAE: {} train_RMSE: {} MSE: {}".format(fin_train_MAE, fin_train_RMSE, loss1))

    test_MAE = []
    test_RMSE = []
    n_test = 0

    model.eval()
    eval_loss = 0.0
    eval_acc = 0.0
    for x, y in valid_loader:
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        x = x.to(device)
        y = y.to(device)
        out = model(x)
        loss = criterion(out, y)
        eval_loss += loss.item()
        MAE, RMSE = test_model(out, y)
        test_MAE.append(MAE)
        test_RMSE.append(RMSE)
        n_test += 1
    loss1 = eval_loss / n_test
    fin_test_MAE = np.sum(test_MAE) / n_test
    fin_test_RMSE = np.sum(test_RMSE) / n_test
    draw_test_MAE.append(fin_test_MAE)
    draw_test_RMSE.append(fin_test_RMSE)
    draw_test_MSE.append(loss1)
    print("test_MAE: {} test_RMSE: {}  MSE: {}".format(fin_test_MAE, fin_test_RMSE, loss1))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10 * 2, 8))

ax1.plot(draw_train_MAE, label='train loss')
ax1.legend()
ax1.set_title("train_MAE")

ax2.plot(draw_test_MAE, label='valid loss')
ax2.legend()
ax2.set_title("test_MAE")

fig.savefig("./MAE.png", bbox_inches='tight')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10 * 2, 8))

ax1.plot(draw_train_RMSE, label='train loss')
ax1.legend()
ax1.set_title("Loss Curve")

ax2.plot(draw_test_RMSE, label='valid loss')
ax2.legend()
ax2.set_title("Loss Curve")

fig.savefig("./RMSE.png", bbox_inches='tight')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10 * 2, 8))

ax1.plot(draw_train_MSE, label='train loss')
ax1.legend()
ax1.set_title("Loss Curve")

ax2.plot(draw_test_MSE, label='valid loss')
ax2.legend()
ax2.set_title("Loss Curve")

fig.savefig("./MSE.png", bbox_inches='tight')