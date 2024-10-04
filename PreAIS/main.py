import numpy as np
from processdata import process_data, extract_xlsx_sequences
from metric import calculate_mcc, calculate_acc_sn_sp
from model.CNN_DNN import model_DNN_CNN
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import os
import random
from sklearn.linear_model import LinearRegression


def set_randomseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, save_path, num_epoch, is_train=True, fold=None):
    os.makedirs(save_path, exist_ok=True)
    if is_train:
        torch.save(model.state_dict(), os.path.join(save_path, f'fold{fold}.pth'))
    else:
        torch.save(model.state_dict(),
                   os.path.join(save_path, f'epochs{num_epoch}.pth'))


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, fold):
    best_loss = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch+1) % 10 == 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                    val_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            if val_loss < best_loss:
                best_loss = val_loss
                save_model(model=model, save_path='ckpt', is_train=True, fold=fold, num_epoch=epoch)
            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


def cross_validation(data, data_labels, k=10):
    kf = KFold(n_splits=k, shuffle=True)
    acc = []
    sn = []
    sp = []
    mcc = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for fold, (train_index, val_index) in enumerate(kf.split(data), 1):
        model = model_DNN_CNN(data.shape[1], 1)
        X_train, X_val = data[train_index], data[val_index]
        y_train, y_val = data_labels[train_index], data_labels[val_index]
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                       torch.tensor(y_train, dtype=torch.float32))
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                                      torch.tensor(y_val, dtype=torch.float32))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1001, shuffle=False)
        model = model.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=5e-2)

        # Training
        train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=500, fold=fold)

        file_list = os.listdir('ckpt')
        files = [file_name for file_name in file_list if file_name.startswith('fold')]
        for file in files:
            if file.startswith(f'fold{fold}'):
                model_path = os.path.join('ckpt', file)
                model = model.to(device)
                model.load_state_dict(torch.load(model_path))
        # Evaluation
        model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs).cpu().numpy()
                predictions.extend(outputs)
                actuals.extend(labels.numpy())

        predictions = np.array(predictions) > 0.5  # Threshold for binary classification
        Accuracy, Sensitivity, Specificity = calculate_acc_sn_sp(actuals, predictions)
        Mcc = calculate_mcc(actuals, predictions)
        print(f'fold:{fold}:acc:{Accuracy:.4f} sp:{Specificity:.4f} sn:{Sensitivity:.4f} mcc:{Mcc:.4f}')
        acc.append(Accuracy)
        sn.append(Sensitivity)
        sp.append(Specificity)
        mcc.append(Mcc)

    print(f'all_fold:acc:{np.mean(acc):.4f} sp:{np.mean(sp):.4f} sn:{np.mean(sn):.4f} mcc:{np.mean(mcc):.4f}')


def train_for_test(model, train_loader, test_data, test_lable, criterion, optimizer, num_epochs):
    best_acc = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_data = torch.tensor(test_data, dtype=torch.float).to(device)
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss/len(train_loader)
        if (epoch+1)%10 == 0:
            save_model(model=model, save_path='ckpt', is_train=False, fold=0, num_epoch=epoch)
            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}')

if __name__ == '__main__':
    # set_randomseed(42)
    file_path1 = "./data/positive_sequences5000.txt"
    file_path2 = "./data/negative_sequences5000.txt"
    data, label = process_data(file_pos_path=file_path1, file_neg_path=file_path2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cross_validation(data, label)


    test_data, labels = extract_xlsx_sequences('./data/test_seq.xlsx')
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(data, dtype=torch.float32),
                                                   torch.tensor(label, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)

    model = model_DNN_CNN(test_data.shape[1], 1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=5e-2)
    train_for_test(model, train_loader, test_data, labels, criterion, optimizer, 50000)

