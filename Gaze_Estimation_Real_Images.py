import torch
import os
import time
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, root_mean_squared_error
from torchvision import models, transforms
from PIL import Image
# import matplotlib.pyplot as plt
import pandas as pd
import wandb

wandb.init("WANDB PROJECT NAME")

learning_rate = 1e-4
batch_size = 32
nof_epochs = 5

img_dir_path = "img_source"
labels_path_train = "label_source"
labels_path_val = "label_source"
model_save_path = "model_path"

ttl_train_time, best_mae = 0, 200
losses_val, maes, rmses, r2s = [], [], [], []

class GazeDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.label_file = label_file
        self.data = []
        for csv in os.listdir(self.label_file):
            csv_path = os.path.join(label_file, csv)
            df = pd.read_csv(csv_path, dtype="float32", converters={'0': str})
            self.data.append(df)
        self.data = pd.concat(self.data, ignore_index=True)
        self.transform = transform
        self.img_files = self.data.iloc[:, 0].tolist()
        self.gaze = self.data.iloc[:, [1, 2]].to_numpy()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                if self.transform is not None:
                    img = self.transform(img)
                    
            rgaze = torch.tensor(self.gaze[idx], dtype=torch.float32)
            return img, rgaze
        
def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)
    
class GazeModel(torch.nn.Module):
    def __init__(self):
        super(GazeModel, self).__init__()
        self.base = models.resnet34(pretrained=True)
        in_features = self.base.fc.in_features

        self.base.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 2))
        
    def forward(self, x):
        return self.base(x)
        
#mean & std values of ImageNet dataset
mean_c, std_c = [0.485,0.456,0.406], [0.229,0.224,0.225]

transform = transforms.Compose([
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_c, std=std_c)])

model = GazeModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")

criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience = 4)

# checkpoint = {
#     "model_state_dict":model.state_dict(),
#     "optimizer_state_dict":optimizer.state_dict(),
#     "scheduler_state_dict":scheduler.state_dict(),
#     "best_mae":best_mae}

if os.path.exists(model_save_path):
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    best_mae = checkpoint["best_mae"]
else:
    model.to(device)
    
# optimizer.param_groups[0]['lr'] = learning_rate

dataset_train = GazeDataset(img_dir_path, labels_path_train, transform=transform)
dataset_val = GazeDataset(img_dir_path, labels_path_val, transform=transform)

data_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
data_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# print(f"Val Data: {len(data_val.dataset)}")
# print(f"Train Data: {len(data_train.dataset)}")

# for i in range(5):
#     img, gazes = data_val.dataset[i]
#     print(i, img.shape, gazes)
    
# for i in range(5):
#     img, gazes = data_train.dataset[i]
#     print(i, img.shape, gazes)

for param in model.parameters():
    param.requires_grad = False

for param in model.base.fc.parameters():
    param.requires_grad = True
    
for param in model.base.layer4.parameters():
    param.requires_grad = True
    
for param in model.base.layer3.parameters():
    param.requires_grad = True
    
# for param in model.base.layer2.parameters():
#     param.requires_grad = True
    
for epoch in range(nof_epochs):
    
    running_loss_train, ttl_samples_train = 0.0, 0
    running_loss_val, ttl_abs_error_val, ttl_samples_val, = 0.0, 0.0, 0
    epoch_gazes_val, epoch_preds_val = [], []
    
    model.train()
    
    start_time = time.time()
    
    for batch in data_train:
        if batch is None: continue
        img, gazes = batch
        img, gazes = img.to(device), gazes.to(device)
        optimizer.zero_grad()
        outputs = model(img)
        loss = criterion(outputs, gazes)
        loss.backward()
        optimizer.step()
        
        running_loss_train += loss.item()
        ttl_samples_train += gazes.size(0)
     
    train_epoch_time = time.time() - start_time
    ttl_train_time += train_epoch_time
    curr_lr = optimizer.param_groups[0]['lr']
    epoch_loss_train = running_loss_train / ttl_samples_train
    
    print(f"\nEpoch: {epoch+1}",
          f"\nTrain Loss: {epoch_loss_train:.2f}",
          f"\nLearning Rate: {curr_lr:.6f}",
          f"\nTrain Time: {train_epoch_time:.2f}")
    
    model.eval()
    
    # start_time = time.time()
    
    with torch.no_grad():
        for batch in data_val:
            if batch is None: continue
            img, gazes = batch
            img, gazes = img.to(device), gazes.to(device)
            outputs = model(img)
            loss = criterion(outputs, gazes)
            
            preds = outputs
            epoch_gazes_val.extend(gazes.detach().cpu().numpy())
            epoch_preds_val.extend(preds.detach().cpu().numpy())
            
            running_loss_val += loss.item()
            ttl_abs_error_val += torch.sum(torch.abs(outputs-gazes)).item()
            ttl_samples_val += gazes.size(0)
            
    scheduler.step(running_loss_val)
    
    # val_epoch_time = time.time() - start_time
    epoch_loss_val = running_loss_val / ttl_samples_val
    epoch_mae_val = ttl_abs_error_val / ttl_samples_val
    r2_scr = r2_score(epoch_gazes_val, epoch_preds_val)
    rmse = root_mean_squared_error(epoch_gazes_val, epoch_preds_val)
    
    # losses_val.append(epoch_loss_val)
    # rmses.append(rmse)
    # maes.append(epoch_mae_val)
    # r2s.append(r2_scr)
    
    # print(f"\nVal Loss: {epoch_loss_val:.2f}",
    #       f"\nMAE: {epoch_mae_val:.2f}",
    #       f"\nR2 Score: {r2_scr:.2f}",
    #       f"\nRMSE: {rmse:.2f}",
    #       f"\nVal Time: {val_epoch_time:.2f}")
    
    if epoch_mae_val < best_mae:
        best_mae = epoch_mae_val
        checkpoint = {
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict":optimizer.state_dict(),
            "scheduler_state_dict":scheduler.state_dict(),
            "best_mae":best_mae}
        torch.save(checkpoint, model_save_path)
        print(f"New best performing model saved at epoch: {epoch + 1} with MAE: {epoch_mae_val:.2f}")
        
    wandb.log({
        "train_loss": epoch_loss_train,
        "val_loss": epoch_loss_val,
        "mae": epoch_mae_val,
        "rmse": rmse,
        "r2": r2_scr,
        "pred": preds,
        "real": gazes,
        "epoch_time": train_epoch_time,
        "learning_rate": curr_lr
    })
    
    wandb.save("MODEL_PATH")

# plt.figure(figsize=(12,5))

# plt.subplot(1,4,1)
# plt.plot(losses_val, label="Losses")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Loss - Epoch")
# plt.legend()

# plt.subplot(1,4,2)
# plt.plot(maes, label="MAE")
# plt.xlabel("Epoch")
# plt.ylabel("MAE")
# plt.title("MAE - Epoch")
# plt.legend()

# plt.subplot(1,4,3)
# plt.plot(rmses, label="RMSE")
# plt.xlabel("Epoch")
# plt.ylabel("RMSE")
# plt.title("RMSE - Epoch")
# plt.legend()

# plt.subplot(1,4,4)
# plt.plot(r2s, label="R2 Scores")
# plt.xlabel("Epoch")
# plt.ylabel("R2 Score")
# plt.title("R2 Score - Epoch")
# plt.legend()

# print(f"\nTotal time passed for training: {ttl_train_time:.2f}")
            







