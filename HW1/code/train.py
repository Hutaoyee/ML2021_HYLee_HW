import math
import numpy as np

# Read write data
# Excel
import pandas as pd
import os
import csv

# 进度条
from tqdm import tqdm

import torch
# 卷积层、Loss等
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

# 可视化
from torch.utils.tensorboard import SummaryWriter

# 设置固定随机种子: 保证每次运行结果一致
def sama_seed(seed):
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        
        torch.cuda.manual_seed_all(seed)
    
# 划分数据集
def train_valid_split(data_set, valid_ratio, seed):
    
    # 划分的长度
    valid_data_size = int(len(data_set) * valid_ratio)
    train_data_size = len(data_set) - valid_data_size
    
    # 划分
    train_data, valid_data = random_split(data_set, [train_data_size, valid_data_size], generator=torch.Generator().manual_seed(seed))
    
    return np.array(train_data), np.array(valid_data)
    
# 选择特征
def select_feat(train_data, valid_data, test_data, select_all = True):
    
    # 选择label
    # 所有行的最后一列
    label_train = train_data[:, -1]
    label_valid = valid_data[:, -1]
    
    # 选择feature
    # 所有行，除了第一列（id）和最后一列（label）
    raw_train = train_data[:, 1:-1]
    raw_valid = valid_data[:, 1:-1]
    # 测试集无label列
    raw_test = test_data[:, 1:]
    
    # 是否选择所有特征
    if select_all:
        
        feat_idx = list(range(raw_train.shape[1]))
        
    else:
        
        feat_idx = [0,]
        
    return raw_train[:, feat_idx], raw_valid[:, feat_idx], raw_test[:, feat_idx], label_train, label_valid

# 设置数据集
# 继承自PyTorch的Dataset类，需要重写__init__、__len__、__getitem__方法
class COVID19Dataset(Dataset):
    
    def __init__(self, features, targets=None):
        
        # 档targets为None时，为预测集，否则为训练集
        # targets为label值？
        if targets is None:
            
            self.targets = targets
            
        else:
            
            self.targets = torch.FloatTensor(targets)
            
        self.features = torch.FloatTensor(features)
        
    # 从数据集取出一组数据
    def __getitem__(self, index):
        
        if self.targets is None:
            
            return self.features[index]
        
        else:
            
            return self.features[index], self.targets[index]
            
    # 返回数据集长度
    def __len__(self):
        
        return len(self.features)

# 构建DNN Model
class My_DNNModel(nn.Module):
    
    def __init__(self, input_dim):
        
        super(My_DNNModel, self).__init__()
        
        self.layers = nn.Sequential(
            
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
    def forward(self, x):
        
        # 讲传入数据向前在layers中传递
        x = self.layers(x)
        # 将传递过程中维度不匹配的维度去除
        x = x.squeeze(1)
        
        return x
        
# 参数设置
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    
    'seed': 5201314,
    'select_all': True,
    'valid_ratio': 0.2,
    # 迭代次数
    'n_epochs': 4000,
    'batch_size': 1024,
    'learning_rate': 1e-5,
    # 当模型400不更新时，停止训练
    'early_stop': 400,
    'model_path': './HW1/models/model.ckpt',
}

# 训练过程
def trainer(train_loader, valid_loader, model, config, device):
    
    # Loss函数
    criterion = nn.MSELoss(reduction='mean')
    # optimizer算法
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
    
    # 可视化
    writer = SummaryWriter()
    
    if not os.path.isdir('./HW1/models'):
        
        os.makedirs('./HW1/models')
        
    # 其他参数设置
    n_epochs = config['n_epochs']
    best_loss = math.inf
    step = 0
    early_stop_count = 0
    
    # 训练实现
    for epoch in range(n_epochs):
        
        # 训练模式
        model.train()
        # loss记录集
        loss_train = []
        # 进度条
        train_pbar = tqdm(train_loader, position=0, leave=True)
        
        # train loop
        # 从 DataLoader 中获取批次数据
        for x, y in train_pbar:
            
            # 梯度清零
            optimizer.zero_grad()
            
            x, y = x.to(device), y.to(device)
            
            # 前向传播与损失计算
            pred = model(x)
            loss = criterion(pred, y)
            
            # 反向传播与优化
            loss.backward()
            optimizer.step()
            
            step += 1
            loss_train.append(loss.detach().item())
            
            # 显示训练过程
            train_pbar.set_description(f'Epoch {epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
        
        # 平均Loss值与绘图
        mean_train_loss = sum(loss_train) / len(loss_train)
        writer.add_scalar('Loss/train', mean_train_loss, step)
        
        # valid loop
        model.eval()
        loss_valid = []
        
        for x, y in valid_loader:
            
            x, y = x.to(device), y.to(device)
            
            with torch.no_grad():
                
                pred = model(x)
                loss = criterion(pred, y)
            
            loss_valid.append(loss.detach().item())
            
        mean_valid_loss = sum(loss_valid) / len(loss_valid)
        print(f"""Epoch {epoch + 1}/{n_epochs}]: Train_loss:{mean_train_loss:.4f}, 
              Valid_loss:{mean_valid_loss:.4f}""") 
        writer.add_scalar('Loss/valid', mean_valid_loss, step)
        
        # 判断最好loss值是否更新
        if mean_valid_loss < best_loss:
            
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['model_path'])
            print('Save model! With Loss: {:.4f}...'.format(best_loss))
            early_stop_count = 0
            
        else:
            
            early_stop_count += 1
            
        if early_stop_count >= config['early_stop']:

            print("\n Model is not improving, so halt training...")
            return
        
# 调用实例、方法对数据进行处理、设置参数
# 种子
same_seed = (config['seed'])

# TODO
# 读取数据
train_data = pd.read_csv('./HW1/covid.train.csv').values
test_data = pd.read_csv('./HW1/covid.test.csv').values

# 划分数据集
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], 
                                            config['seed'])
print(f"""Train data size: {train_data.shape}, Valid data size: {valid_data.shape}, 
      Test data size: {test_data.shape}""")

# 选择特征
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, 
                                                          test_data, config['select_all'])
print(f"the number of features : {x_train.shape[1]}")

# 构造数据集
train_dataset = COVID19Dataset(x_train, y_train)
valid_dataset = COVID19Dataset(x_valid, y_valid)
test_dataset = COVID19Dataset(x_test)

# 迭代器(Dataloader)：多线程读取数据，组成batch
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
        
# START!!!
model = My_DNNModel(input_dim = x_train.shape[1]).to(device)
trainer(train_loader, valid_loader, model, config, device)

# 预测值检验
def predict(test_loader, model, device):
    
    model.eval()
    preds = []
    
    for x in tqdm(test_loader):
        
        x = x.to(device)
        
        with torch.no_grad():
            
            pred = model(x)
            preds.append(pred.detach().cpu())
            
        preds = torch.cat(preds, dim=0).numpy()
        
        return preds
    
def save_pred(preds, file):
    
    with open(file, 'w') as fp:
        
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        
        for i, p in enumerate(preds):
            
            writer.writerow([i, p])

# 保存结果
model = My_DNNModel(input_dim = x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['model_path']))
preds = predict(test_loader, model, device)

if not os.path.exists('./HW1/preds'):
    
    os.makedirs('./HW1/preds')
    
save_pred(preds, './HW1/preds/pred.csv')