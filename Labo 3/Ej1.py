# -*- coding: utf-8 -*-
# %pip install -q torch_snippets
import torch
import modelo
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch_snippets import *

################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
################################################################
def train_batch(input, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, input)
    loss.backward()
    optimizer.step()
    return loss

@torch.no_grad()
def validate_batch(input, model, criterion):
    model.eval()
    output = model(input)
    loss = criterion(output, input)
    return loss
################################################################
img_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

data_root = '../Data/BaseOCR_MultiStyle'
full_ds = datasets.ImageFolder(root=data_root, transform=img_transform)

val_len = int(0.1 * len(full_ds))
trn_len = len(full_ds) - val_len
trn_ds, val_ds = torch.utils.data.random_split(full_ds, [trn_len, val_len])

# DataLoaders
batch_size = 32
trn_dl = DataLoader(dataset=trn_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False)

# instanciar modelo
model = modelo.AutoEncoder(10).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

num_epochs = 5

train_loss = []
val_loss = []
for epoch in range(num_epochs):
    N = len(trn_dl)
    tloss = 0.0
    for ix, (data, _) in enumerate(trn_dl):
        data = data.to(device)
        loss = train_batch(data, model, criterion, optimizer)
        tloss += loss.item()
    train_loss.append(tloss / N)

    N = len(val_dl)
    vloss = 0.0
    for ix, (data, _) in enumerate(val_dl):
        data = data.to(device)
        loss = validate_batch(data, model, criterion)
        vloss += loss.item()
    val_loss.append(vloss / N)    

####################################################################
# graficar las losses de entrenamiento y de validacion
if len(train_loss) > 0:
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o', label='train')
    plt.plot(range(1, len(val_loss) + 1), val_loss, marker='o', label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
####################################################################

for _ in range(3):
    ix = np.random.randint(len(val_ds))
    im, _ = val_ds[ix]
    im_coll = im.unsqueeze(0).to(device)
    with torch.no_grad():
        _im = model(im_coll)[0].cpu()
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    show(im[0].cpu(), ax=ax[0], title='input')
    show(_im[0], ax=ax[1], title='prediction')
    plt.tight_layout()
    plt.show()