import torch
import numpy.random as npr
import os
import argparse
import logging
import time
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
from model.config_hyper_para import *
from data_prepro import *
from model.models import *
import model.regular as regul
import math
from model.formers import GRUModel_former3
from model.utils import NeuralFourierLayer, CalculateMubo
import matplotlib
matplotlib.use('agg')

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', '-a', type=eval, default=False)
parser.add_argument('--dataset', '-d', help = "dataset", type = str)
parser.add_argument('--train_dir', type=str, default=None)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

data_name = 'pm25'
if 'energy' in data_name:
    data_path = 'raw_data/energydata_complete.csv'
elif 'nasdaq' in data_name:
    data_path = 'raw_data/nasdaq100_padding.csv'
elif 'pm25' in data_name:
    data_path = 'raw_data/pm25_rawdata.csv'
elif 'sml' in data_name:
    data_path = 'raw_data/sml_rawdata.csv'
elif 'elec' in data_name:
    data_path = 'raw_data/electricity_rawdata.csv'
else:
    print("wrong dataset name")

args.train_dir = 'arbitrary-step/' + data_name
print('current datset: {}'.format(data_name))
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

target_col = target_col_dic[data_name]
indep_col = indep_col_dic[data_name]
win_size = 20
pre_T = 5
norm = True
alpha_loss = 1

train_share = 0.8
is_stateful = False
normalize_pattern = 2
generator = getGenerator(data_path)
datagen = generator(data_path, target_col, indep_col, win_size, pre_T,
                    train_share, is_stateful, normalize_pattern)

train_x, train_y, val_x, val_y, test_x, test_y, y_mean, y_std = datagen.data_extract_val_arbi()

ystd = torch.Tensor([y_std]).to(device)
ymean = torch.Tensor([y_mean]).to(device)

torch.manual_seed(117)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(117)

print(" --- Data shapes: ", np.shape(train_x), np.shape(train_y), np.shape(val_x), np.shape(val_y), np.shape(test_x), np.shape(test_y))

batch_size = 32
dataset_train = subDataset(train_x, train_y)
dataset_val = subDataset(val_x, val_y)
dataset_test = subDataset(test_x, test_y)
train_loader = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, num_workers=1,drop_last=True)
val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)
test_loader = DataLoader(dataset_test, batch_size=batch_size,shuffle=False, num_workers=1,drop_last=True)


def train(train_loader):
    loss_list = []
    loss_mse = []
    loss_kl = []
    for i, (inputs, target) in enumerate(train_loader):

        inputs = Variable(inputs.view(-1, step, input_dim).to(device))
        target = Variable(target.to(device))

        # 编码
        h, _, _ = encoder(inputs)  # i:[128, 20, 16] o:list[128, 320]
        att_h_input = torch.stack(h, dim=1)  # o:[128, 20, 320]

        h2, _, _ = encoder2(inputs)  # i:[128, 20, 16] o:list[128, 320]
        att_h_input2 = torch.stack(h2, dim=1)  # o:[128, 20, 320]

        qz_para = att(att_h_input, att_h_input2)  # [batch_size, 2*time_step]
        qz_para2 = att2(att_h_input2, att_h_input)  # [128, 32]

        qz0_mean, qz0_logvar = qz_para[:, :input_dim], qz_para[:, input_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(device)
        z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean

        qz1_mean, qz1_logvar = qz_para2[:, :input_dim], qz_para2[:, input_dim:]
        epsilon1 = torch.randn(qz1_mean.size()).to(device)
        z1 = epsilon1 * torch.exp(0.5 * qz1_logvar) + qz1_mean

        # forward in time and solve ode for reconstructions
        pred_z = odeint(func, z0, train_T).permute(1, 0, 2)  # delta_h, 初始值，时间
        pred_z1 = odeint(func2, z1, train_T).permute(1, 0, 2)
        if norm:
            pred_0 = fc(pred_z).squeeze(2)
            pred_y0 = pred_0 * ystd + ymean
            pred_1 = fc2(FourierNet(pred_z1))[:, -pre_T:, :].squeeze(2)
            pred_y1 = pred_1
            pred_y = pred_y0 + pred_y1

        # Calculate Loss
        noise_std_ = torch.zeros(pred_y.size()).to(device) + noise_std
        noise_logvar = 2. * torch.log(noise_std_).to(device)
        logpx = log_normal_pdf(target, pred_y, noise_logvar).sum(-1)
        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar, pz0_mean, pz0_logvar).sum(-1)
        pz1_mean = pz1_logvar = torch.zeros(z1.size()).to(device)
        analytic_kl2 = normal_kl(qz1_mean, qz1_logvar, pz1_mean, pz1_logvar).sum(-1)

        l_px = torch.mean(-logpx, dim=0)
        l_kl = -torch.mean(analytic_kl, dim=0)
        l_kl2 = -torch.mean(analytic_kl2, dim=0)
        loss2 = criterion(target, pred_y)

        loss = alpha_loss * (l_px + l_kl) + loss2 + 0.0001*l_kl2

        if torch.cuda.is_available():
            loss.to(device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_kl.append(l_kl.item())
        loss_mse.append(loss2.item())
        loss_list.append(loss.item())

    train_mse = np.array(loss_mse).mean()
    kl_loss = np.array(loss_kl).mean()
    total_loss = np.array(loss_list).mean()
    return total_loss, kl_loss, train_mse


def eval(x, y):
    # Calculate Accuracy
    total = 0
    mae = []
    rmse = []
    # Iterate through test dataset
    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = Variable(torch.Tensor(x).to(device))
            target = Variable(torch.Tensor(y).to(device))
        else:
            inputs = Variable(x)

        # decoder input
        h, _, _ = encoder(inputs)  # 源代码
        att_h_input = torch.stack(h, dim=1)

        h2, _, _ = encoder2(inputs)  # 源代码
        att_h_input2 = torch.stack(h2, dim=1)

        qz_para = att(att_h_input, att_h_input2)
        qz_para2 = att2(att_h_input2, att_h_input)

        qz0_mean, qz0_logvar = qz_para[:, :input_dim], qz_para[:, input_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(device)
        z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
        qz1_mean, qz1_logvar = qz_para2[:, :input_dim], qz_para2[:, input_dim:]
        epsilon = torch.randn(qz1_mean.size()).to(device)
        z1 = epsilon * torch.exp(0.5 * qz1_logvar) + qz1_mean

        # forward in time and solve ode for reconstructions
        pred_z = odeint(func, z0, test_T).permute(1, 0, 2)
        pred_z1 = odeint(func2, z1, test_T).permute(1, 0, 2)
        if norm:
            pred_0 = fc(pred_z).squeeze(2)
            pred_y0 = pred_0 * ystd + ymean
            pred_1 = fc2(FourierNet(pred_z1))[:, -pre_T:, :].squeeze(2)
            pred_y1 = pred_1
            pred_y = pred_y0 + pred_y1

        test_mse = criterion(target, pred_y)
        test_rmse = torch.sqrt(test_mse)
        test_mae = criterionL1(target, pred_y)

        return test_mae, test_rmse


def log_test(text_env, errors):
    text_env.write("\n testing error: %s \n\n" % (errors))
    return


def test_part(x, y):
    # Calculate Accuracy
    # Iterate through test dataset
    with torch.no_grad():
        if torch.cuda.is_available():
            inputs = Variable(torch.Tensor(x).to(device))
            target = Variable(torch.Tensor(y).to(device))
        else:
            inputs = Variable(x)

        # decoder input
        h, _, _ = encoder(inputs)  # 源代码
        att_h_input = torch.stack(h, dim=1)

        h2, _, _ = encoder2(inputs)  # 源代码
        att_h_input2 = torch.stack(h2, dim=1)

        qz_para = att(att_h_input, att_h_input2)
        qz_para2 = att2(att_h_input2, att_h_input)

        qz0_mean, qz0_logvar = qz_para[:, :input_dim], qz_para[:, input_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(device)
        z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
        qz1_mean, qz1_logvar = qz_para2[:, :input_dim], qz_para2[:, input_dim:]
        epsilon = torch.randn(qz1_mean.size()).to(device)
        z1 = epsilon * torch.exp(0.5 * qz1_logvar) + qz1_mean

        # forward in time and solve ode for reconstructions
        pred_z = odeint(func, z0, test_T).permute(1, 0, 2)
        pred_z1 = odeint(func2, z1, test_T).permute(1, 0, 2)
        if norm:
            pred_0 = fc(pred_z).squeeze(2)
            pred_y0 = pred_0 * ystd + ymean
            pred_1 = fc2(FourierNet(pred_z1))[:, -pre_T:, :].squeeze(2)
            pred_y1 = pred_1
            pred_y = pred_y0 + pred_y1

        test_mse = criterion(target, pred_y)
        test_rmse = torch.sqrt(test_mse)
        test_mae = criterionL1(target, pred_y)
        test_mse1 = criterion(target[:,[0]], pred_y[:,[0]])
        test_rmse1 = torch.sqrt(test_mse1)
        test_mae1 = criterionL1(target[:,[0]], pred_y[:,[0]])
        test_mse2 = criterion(target[:,[1]], pred_y[:,[1]])
        test_rmse2 = torch.sqrt(test_mse2)
        test_mae2 = criterionL1(target[:,[1]], pred_y[:,[1]])
        test_mse3 = criterion(target[:,[2]], pred_y[:,[2]])
        test_rmse3 = torch.sqrt(test_mse3)
        test_mae3 = criterionL1(target[:,[2]], pred_y[:,[2]])
        test_mse4 = criterion(target[:,[3]], pred_y[:,[3]])
        test_rmse4 = torch.sqrt(test_mse4)
        test_mae4 = criterionL1(target[:,[3]], pred_y[:,[3]])
        test_mse5 = criterion(target[:,[4]], pred_y[:,[4]])
        test_rmse5 = torch.sqrt(test_mse5)
        test_mae5 = criterionL1(target[:,[4]], pred_y[:,[4]])

        return test_mae, test_rmse, test_mae1, test_rmse1, test_mae2, test_rmse2, test_mae3, test_rmse3, test_mae4, \
               test_rmse4, test_mae5, test_rmse5, pred_y

input_dim = train_x.shape[-1]
pre_step = train_y.shape[-1]
step = train_x.shape[1]
target_size = train_y.shape[1]

train_T = torch.Tensor([1., 1.5, 2., 2.5, 3.]).to(device)  # torch.linspace(1., 3., 3).to(device)
test_T = torch.Tensor([1., 1.5, 2., 2.5, 3.]).to(device)

num_epochs = 100
temp_attention_type = 'temp_loc'
vari_attention_type = 'vari_loc_all'
hidden_dim = hidden_dim_dic[data_name]
qz_para_dim = 10
layer_dim = 1  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
gate_type = 'tensor'
loss_type = 'mse'
activation_type = ''
criterion = torch.nn.MSELoss()
criterionL1 = torch.nn.L1Loss()
learning_rate = lr_dic[data_name]
weight_decay = 0.01
noise_std = 0.1
per_vari_dim = int(hidden_dim / input_dim)
path_home = './arbitrary-step/'
path = path_home + data_name + '/'

num = 201
best_log_dir = path + data_name + '_ode_att_model_' + str(pre_T) + '-' + str(num) + '.pth'
log_err_file = path + data_name + '_test.txt'

# encoder = GRUModel(hidden_dim, input_dim, layer_dim, gate_type, qz_para_dim).to(device)
# func = ODEfunc(input_dim).to(device)
# att = attention2_mix(hidden_dim, input_dim, step, pre_step, per_vari_dim, temp_attention_type,
#                         vari_attention_type).to(device)
# fc = nn.Linear(input_dim, 1).to(device)

encoder = GRUModel(hidden_dim, input_dim, layer_dim, gate_type, qz_para_dim).to(device)
func = ODEfunc(input_dim).to(device)
att = GRUModel_former3(hidden_dim, input_dim, layer_dim, gate_type, qz_para_dim).to(device)
fc = nn.Linear(input_dim, 1).to(device)

encoder2 = GRUModel(hidden_dim, input_dim, layer_dim, gate_type, qz_para_dim).to(device)
func2 = ODEfunc(input_dim).to(device)
att2 = GRUModel_former3(hidden_dim, input_dim, layer_dim, gate_type, qz_para_dim).to(device)
fc2 = nn.Linear(input_dim, 1).to(device)
inner_s = 128
out_dim = 1
FourierNet = NeuralFourierLayer(inner_s, out_dim, win_size, pre_T).to(device)

params = (list(encoder.parameters()) + list(func.parameters()) + list(att.parameters()) +
          list(fc.parameters()) + list(encoder2.parameters()) + list(func2.parameters()) +
          list(att2.parameters()) + list(FourierNet.parameters()) + list(fc2.parameters()))

optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
loss_meter = RunningAverageMeter()

best_val_mae = float("inf")
best_val_rmse = float("inf")
test_mae = 0
test_rmse = 0
iter = 0
start_epoch = 0

if args.train_dir is not None:
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    ckpt_path = os.path.join(args.train_dir, 'ckpt_' + data_name + str(pre_T) + '-' + str(num) +'.pth')
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        func.load_state_dict(checkpoint['func_state_dict'])
        att.load_state_dict(checkpoint['att_state_dict'])
        fc.load_state_dict(checkpoint['fc_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print('Loaded ckpt from epoc {}'.format(start_epoch))
    else:
        start_epoch = 0
        print('no saved model, start from epoc 1')


for epoch in range(start_epoch + 1, num_epochs + 1):
    train_loss, kl_loss, mse_loss = train(train_loader)
    val_mae, val_rmse = eval(val_x, val_y)
    print('Epoch: {}. Loss: {}. Loss_kl: {}. Loss_mse: {}'.format(epoch, train_loss, kl_loss, mse_loss))

    total_mae, total_rmse, mae_1, rmse_1, mae_2, rmse_2, mae_3, rmse_3, mae_4, rmse_4, mae_5, rmse_5, prediction = test_part(
        test_x, test_y)
    print(
        'test_1_mae: {} test_1_rmse: {} test_2_mae: {} test_2_rmse:{} test_3_mae: {} test_3_rmse: {} test_4_mae: {} test_4_rmse: {} test_5_mae: {} test_5_rmse: {}'.format(
            mae_1, rmse_1, mae_2, rmse_2, mae_3, rmse_3, mae_4, rmse_4, mae_5, rmse_5))
    save = prediction.cpu().detach().numpy()
    np.savetxt(path + 'prediction2.csv', save, delimiter=',')
    if val_rmse < best_val_rmse:
        best_val_mae = val_mae
        best_val_rmse = val_rmse
        test_mae, test_rmse = eval(test_x, test_y)

        state = {'encoder_state_dict':encoder.state_dict(),
                 'func_state_dict':func.state_dict(),
                 'att_state_dict':att.state_dict(),
                 'fc_state_dict':fc.state_dict(),
                 'optimizer_state_dict':optimizer.state_dict(),
                 'epoch':epoch}
        torch.save(state, best_log_dir)
        print('best_val_mae: {}. best_val_rmse: {}'.format(best_val_mae, best_val_rmse))

print('best_val_mae: {}. best_val_rmse: {}'.format(best_val_mae, best_val_rmse))


with open(log_err_file, "a") as text_file:
    log_test(text_file, [best_val_rmse, best_val_mae, batch_size, hidden_dim, learning_rate, weight_decay, noise_std])

test_path = os.path.join(args.train_dir, best_log_dir)

total_mae, total_rmse, mae_1, rmse_1, mae_2, rmse_2, mae_3, rmse_3, mae_4, rmse_4, mae_5, rmse_5, prediction = test_part(test_x, test_y)
save = prediction.cpu().detach().numpy()
np.savetxt(path + 'prediction.csv', save, delimiter=',')
np.savetxt(path + 'target.csv', test_y, delimiter=',')

with open(log_err_file, "a") as text_file:
    log_test(text_file, [total_mae, total_rmse, mae_1, rmse_1, mae_2, rmse_2, mae_3, rmse_3, mae_4, rmse_4, mae_5, rmse_5])