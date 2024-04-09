#!/usr/bin/env python
# coding: utf-8

# # **EEG Motor Imagery Classification Using CNN, Transformer, and MLP**

# ## **Important Libraries**

# In[1]:


import mne
# from mne.io import concatenate_raws

import os
# import re
# import io
import cv2
# import random
# import string
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

# from google.colab.patches import cv2_imshow
from cv2_plt_imshow import cv2_plt_imshow as cv2_imshow

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from torchmetrics.classification import Accuracy

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['axes.facecolor'] = 'lightgray'

warnings.filterwarnings("ignore")
print(torch.__version__)


# ## **Dataset**

# In[2]:


global_path = ''
dir_path_to_save_data = 'processed_data_original'
seed_value = 42


EEG_CHANNEL = 64
# CLASSES = ['left_fist', 'right_fist', 'baseline_open', 'both_feet', 'both_fist']
CLASSES = ['left_fist', 'right_fist', 'baseline_open', 'both_feet']
CLASSES_ID = {'left_fist': 0, 'right_fist': 1, 'baseline_open': 2, 'both_feet': 3}
no_of_classes = len(CLASSES)

all_files = {}
for dir in CLASSES:
    all_files[dir] = [file for file in os.listdir(os.path.join(dir_path_to_save_data, dir)) if file.endswith('.fif')]
    print(f'{dir} has {len(all_files[dir])} files')


# In[3]:


def read_one_subject_file(file_path, id): # return data in milli volts (input data is in micro volts)
    epochs_eeg = mne.read_epochs(os.path.join(dir_path_to_save_data, file_path))
    # print(np.shape(temp))
    eeg_data = epochs_eeg.get_data(copy=False) * 1e-3
    eeg_data = np.array(eeg_data, dtype=np.float32) 
    
    # print(np.shape(eeg_data))

    epoch = np.shape(eeg_data)[0]
    one_action_labels = np.array([id]*epoch)
    one_action_labels = np.expand_dims(one_action_labels, axis=1)
    return eeg_data, one_action_labels

eeg_data, one_action_labels = read_one_subject_file(os.path.join(CLASSES[0], all_files[CLASSES[0]][0]), CLASSES_ID[CLASSES[0]])
print(np.shape(eeg_data), np.shape(one_action_labels))
print(np.min(eeg_data), np.max(eeg_data)) # In Old Implimentation ((3377, 64, 497), -0.698, 1)
eeg_data.dtype


# In[4]:


def get_data(type):
    if type not in ['train', 'valid', 'test']:  
        raise ValueError('type should be either train, valid or test')
    
    dataset = list()
    labels = list()
    c=0
    for dir in CLASSES:
        for file_name in all_files[dir]:
            c+=1
            num = int(file_name.split('-')[0])
            file_path = os.path.join(dir, file_name)
            id = CLASSES_ID[dir]
            eeg_data, one_action_labels = read_one_subject_file(file_path, id)
            if(type == 'train' and  num < 88):
                dataset.append(eeg_data)
                labels.append(one_action_labels)

            elif(type == 'valid' and  num >= 88 and num<98):
                dataset.append(eeg_data)
                labels.append(one_action_labels)

            elif(type == 'test' and num >= 98):
                dataset.append(eeg_data)
                labels.append(one_action_labels)


    final_data = np.vstack(dataset)
    final_data = np.vstack(dataset)
    final_labels = np.squeeze(np.vstack(labels))

    print(c)
    return final_data, final_labels


# In[5]:


import numpy as np

# Generate sample data
array_3d = np.random.rand(3, 4, 5)  # Example 3D array of shape (2, 3, 4)
array_1d = np.arange(3)  # Example 1D array of shape (8,)

print(array_3d)
print(array_1d)


# In[6]:


x_train, y_train = get_data('train')
x_valid, y_valid = get_data('valid')
x_test, y_test = get_data('test')

# join x_train, x_valid, x_test
x = np.concatenate((x_train, x_valid), axis=0)
x = np.concatenate((x, x_test), axis=0)


# In[7]:


# # Assuming train_data and train_labels are your input data and corresponding labels
# train_data_shape = X.shape

# # Generate indices for shuffling
# np.random.seed(seed_value)

# indices = np.arange(train_data_shape[0])
# np.random.shuffle(indices)

# # Shuffle train_data and train_labels using the same indices
# X = X[indices]
# y = y[indices]


# In[8]:


def dataset_info(X, y):
    print(f'Data Shape: {X.shape}, Labels Shape: {y.shape}')
    print(f'Min and Max of Data ({np.min(X)}, {np.max(X)})')
    print(f'Min and Max of labels ({np.min(y)}, {np.max(y)})')


dataset_info(x_train, y_train)
dataset_info(x_valid, y_valid)
dataset_info(x_test, y_test)


# In[9]:


# for i in range(len(y_train)):
#     print(y_train[i])


# In[10]:


class EEGDataset(data.Dataset):
    def __init__(self, x, x_train, x_valid, x_test, y_train=None, y_valid=None, y_test=None, inference=False):
        super().__init__()

        N_SAMPLE = x.shape[0]
        
        if not inference:
            self.train_ds = {
                'x': x_train,
                'y': y_train,
            }
            # print(self.train_ds['x'].shape)
            
            self.val_ds = {
                'x': x_valid,
                'y': y_valid,
            }
            # print(self.val_ds['x'].shape)
            
            self.test_ds = {
                'x': x_test,
                'y': y_test,
            }
            # print(self.test_ds['x'].shape)
        else:
            self.__split = "inference"
            self.inference_ds = {
                'x': [x],
            }

    def __len__(self):
        return len(self.dataset['x'])

    def __getitem__(self, idx):

        x = self.dataset['x'][idx]
        if self.__split != "inference":
            y = self.dataset['y'][idx]
            x = torch.tensor(x).float()
            # y = torch.tensor(y).unsqueeze(-1).float()
            y = torch.tensor(y).float()
            return x, y
        else:
            x = torch.tensor(x).float()
            return x

    def split(self, __split):
        self.__split = __split
        return self

    @classmethod
    def inference_dataset(cls, x):
        return cls(x, inference=True)

    @property
    def dataset(self):
        assert self.__split is not None, "Please specify the split of dataset!"

        if self.__split == "train":
            return self.train_ds
        elif self.__split == "val":
            return self.val_ds
        elif self.__split == "test":
            return self.test_ds
        elif self.__split == "inference":
            return self.inference_ds
        else:
            raise TypeError("Unknown type of split!")


# In[11]:


eeg_dataset = EEGDataset(x=x, x_train=x_train, x_valid=x_valid, x_test=x_test, y_train=y_train, y_valid=y_valid, y_test=y_test)


# In[12]:


# plt.plot(X[18:21, 0, :].T)
# plt.title("Exemplar single-trial epoched data, for electrode 0")
# plt.ylabel("V")
# plt.xlabel("Epoched Sample")
# plt.show()
# plt.clf()


# ## **Model**

# ### **Utils**

# In[13]:


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.losses = []

    def update(self, val):
        self.losses.append(val)

    def show(self):
        out = torch.mean(
            torch.stack(
                self.losses[np.maximum(len(self.losses)-self.num, 0):]
            )
        )
        return out


# ### **Wrapper**

# In[14]:


class ModelWrapper(L.LightningModule):
    def __init__(self, arch, dataset, batch_size, lr, max_epoch):
        super().__init__()

        self.arch = arch
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.max_epoch = max_epoch

        # self.train_accuracy = Accuracy(task="binary") # change
        # self.val_accuracy = Accuracy(task="binary")
        # self.test_accuracy = Accuracy(task="binary")
        
        
        self.train_accuracy = Accuracy(task="multiclass", num_classes=4)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=4)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=4)

        self.automatic_optimization = False

        self.train_loss = []
        self.val_loss = []

        self.train_acc = []
        self.val_acc = []

        self.train_loss_recorder = AvgMeter()
        self.val_loss_recorder = AvgMeter()

        self.train_acc_recorder = AvgMeter()
        self.val_acc_recorder = AvgMeter()


    def forward(self, x):
        return self.arch(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y=y.to(torch.int64)
        y_hat = self(x)
        # loss = F.binary_cross_entropy_with_logits(y_hat, y) # change
        loss = F.cross_entropy(y_hat, y) # change
        # loss = nn.CrossEntropyLoss()(y_hat, y)
        self.train_accuracy.update(y_hat, y)
        acc = self.train_accuracy.compute().data.cpu()
        # print(loss, acc) # temp
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.train_loss_recorder.update(loss.data)
        self.train_acc_recorder.update(acc)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        sch.step()

        self.train_loss.append(self.train_loss_recorder.show().data.cpu().numpy())
        self.train_loss_recorder = AvgMeter()

        self.train_acc.append(self.train_acc_recorder.show().data.cpu().numpy())
        self.train_acc_recorder = AvgMeter()

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y=y.to(torch.int64)
        y_hat = self(x)
        # loss = F.binary_cross_entropy_with_logits(y_hat, y) # change
        loss = F.cross_entropy(y_hat, y) # change
        # loss = nn.CrossEntropyLoss()(y_hat, y)

        self.val_accuracy.update(y_hat, y)
        acc = self.val_accuracy.compute().data.cpu()

        self.val_loss_recorder.update(loss.data)
        self.val_acc_recorder.update(acc)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def on_validation_epoch_end(self):
        self.val_loss.append(self.val_loss_recorder.show().data.cpu().numpy())
        self.val_loss_recorder = AvgMeter()

        self.val_acc.append(self.val_acc_recorder.show().data.cpu().numpy())
        self.val_acc_recorder = AvgMeter()

    def test_step(self, batch, batch_nb):
        x, y = batch
        y=y.to(torch.int64)
        y_hat = self(x)
        # loss = F.binary_cross_entropy_with_logits(y_hat, y) # change
        loss = F.cross_entropy(y_hat, y) # change
        # loss = nn.CrossEntropyLoss()(y_hat, y)
        self.test_accuracy.update(y_hat, y)

        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_acc",
            self.test_accuracy.compute(),
            prog_bar=True,
            logger=True,
        )

    def on_train_end(self):
        # Loss 
        loss_img_file = "content/loss_plot.png"
        plt.plot(self.train_loss, color = 'r', label='train')
        plt.plot(self.val_loss, color = 'b', label='validation')
        plt.title("Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(loss_img_file)
        plt.clf()
        img = cv2.imread(loss_img_file)
        cv2_imshow(img)
        plt.show()

        # Accuracy
        acc_img_file = "content/acc_plot.png"
        plt.plot(self.train_acc, color = 'r', label='train')
        plt.plot(self.val_acc, color = 'b', label='validation')
        plt.title("Accuracy Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()
        plt.savefig(acc_img_file)
        plt.clf()
        img = cv2.imread(acc_img_file)
        cv2_imshow(img)
        plt.show()

    def train_dataloader(self):
        return data.DataLoader(
            dataset=self.dataset.split("train"),
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            dataset=self.dataset.split("val"),
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return data.DataLoader(
            dataset=self.dataset.split("test"),
            batch_size=1,
            shuffle=False,
        )

    def configure_optimizers(self):

        optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
        )
        lr_scheduler = {
            "scheduler": optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    int(self.max_epoch * 0.25),
                    int(self.max_epoch * 0.5),
                    int(self.max_epoch * 0.75),
                ],
                gamma=0.1
            ),
            "name": "lr_scheduler",
        }
        return [optimizer], [lr_scheduler]


# ### **EEG Classification Model**

# In[15]:


class PositionalEncoding(nn.Module):
    """Positional encoding.
    https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html
    """
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.p = torch.zeros((1, max_len, num_hiddens))
        x = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.p[:, :, 0::2] = torch.sin(x)
        self.p[:, :, 1::2] = torch.cos(x)

    def forward(self, x):
        x = x + self.p[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)


# In[16]:


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
        )

        self.layernorm0 = nn.LayerNorm(embed_dim)
        self.layernorm1 = nn.LayerNorm(embed_dim)

        self.dropout = dropout

    def forward(self, x):
        y, att = self.attention(x, x, x)
        y = F.dropout(y, self.dropout, training=self.training)
        x = self.layernorm0(x + y)
        y = self.mlp(x)
        y = F.dropout(y, self.dropout, training=self.training)
        x = self.layernorm1(x + y)
        return x


# In[17]:


class EEGClassificationModel(nn.Module):
    def __init__(self, eeg_channel, dropout=0.1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                eeg_channel, eeg_channel, 11, 1, padding=5, bias=False
            ),
            nn.BatchNorm1d(eeg_channel),
            nn.ReLU(True),
            nn.Dropout1d(dropout),
            nn.Conv1d(
                eeg_channel, eeg_channel * 2, 11, 1, padding=5, bias=False
            ),
            nn.BatchNorm1d(eeg_channel * 2),
        )

        self.transformer = nn.Sequential(
            PositionalEncoding(eeg_channel * 2, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
        )

        self.mlp = nn.Sequential(
            nn.Linear(eeg_channel * 2, eeg_channel // 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            # nn.Linear(eeg_channel // 2, 1), # change
            nn.Linear(eeg_channel // 2, 4), 
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = x.mean(dim=-1)
        x = self.mlp(x)
        return x


# In[18]:


MODEL_NAME = "EEGClassificationModel"
model = EEGClassificationModel(eeg_channel=EEG_CHANNEL, dropout=0.125)


# In[19]:


# import torch
# from torchviz import make_dot

# # Generate random data
# x = torch.randn(12, 64, 497)
# print(f'Input Shape {x.shape}')

# # Print shape


# # generate predictions for the sample data
# y = model(x)
# print(f'Output Shape {y.shape}')

# # generate a model architecture visualization
# make_dot(y,
#          params=dict(model.named_parameters()),
#          show_attrs=True,
#          show_saved=True).render("MyPyTorchModel_torchviz", format="png")


# ## **Training**

# In[20]:


MAX_EPOCH = 100
BATCH_SIZE = 32
LR = 5e-4
CHECKPOINT_DIR = os.getcwd()
# SEED = int(np.random.randint(2147483647))
SEED = 141352557

print(f"Random seed: {SEED}")

model_w = ModelWrapper(model, eeg_dataset, BATCH_SIZE, LR, MAX_EPOCH)

get_ipython().system('rm -rf logs/')


# In[21]:


# %reload_ext tensorboard
# %tensorboard --logdir=logs/lightning_logs/


# In[22]:


tensorboardlogger = TensorBoardLogger(save_dir="logs/")
csvlogger = CSVLogger(save_dir="logs/")
lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint = ModelCheckpoint(
    monitor='val_acc',
    dirpath=CHECKPOINT_DIR,
    mode='max',
)

# early_stopping = EarlyStopping(
#     monitor="val_acc", min_delta=0.00, patience=3, verbose=False, mode="max"
# )


seed_everything(SEED, workers=True)


trainer = Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=MAX_EPOCH,
    logger=[tensorboardlogger, csvlogger],
    # callbacks=[lr_monitor, checkpoint, early_stopping],
    callbacks=[lr_monitor, checkpoint], # exp
    log_every_n_steps=5,
)


# In[26]:


torch.backends.mps.is_available()


# In[24]:


print('Before', model_w.device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model_w.to(device)
print('After', model_w.device)


# In[ ]:


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,3"  # specify which GPU(s) to be used
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"  # specify which GPU(s) to be used


# In[ ]:


# model_w = ModelWrapper.load_from_checkpoint(checkpoint_path="epoch=19-step=19020.ckpt", arch=EEGClassificationModel(eeg_channel=EEG_CHANNEL, dropout=0.125), dataset=eeg_dataset, batch_size=BATCH_SIZE, lr=LR, max_epoch=MAX_EPOCH)


# In[ ]:


# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.
# device


# In[ ]:


# model_w= nn.DataParallel(model_w, device_ids = [1, 3])
# model_w= nn.parallel.DistributedDataParallel(model_w, device_ids = [1, 3])
# model_w.to(device)


# In[25]:


trainer.fit(model_w)


# ## **Testing**

# In[ ]:


# trainer.test(ckpt_path="EEGClassificationModel_best.ckpt")
# trainer.test(model=model_w ,ckpt_path="epoch=6-step=1666.ckpt")


# os.rename(
#     checkpoint.best_model_path,
#     os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.ckpt")
# )


# ## **Inference**

# In[ ]:


def prediction(sample: np.ndarray, model_wrappr: L.LightningModule) -> list:  #(samples_count, EEG_CHANNEL, Data_Points)
    if(sample.shape == 2):
        sample = np.expand_dims(sample, 0)
    sample  = torch.from_numpy(sample)
    print(f'Input Size: {np.shape(sample)}') # (samples_count, EEG_CHANNEL, Data_Points)
    trainer = Trainer()
    pred = trainer.predict(model=model_wrappr, 
                        dataloaders=data.DataLoader(
                        dataset = sample,
                        batch_size=1,
                        shuffle=False,
                    )
            )
    pred = torch.tensor(np.array(pred))
    print(f'Output Size: {np.shape(pred)}') # (samples_count, 1, classes)
    pred = torch.softmax(pred, dim=2)
    # print(pred)
    predicted_class = torch.squeeze(pred.argmax(dim=2))
    return predicted_class


# In[ ]:


eeg_dataset_test = eeg_dataset.split("test")
print(np.shape(eeg_dataset_test.dataset['x']))
# eeg_dataset_test.dataset['x']
# eeg_dataset_test.dataset['y']

y_true = eeg_dataset_test.dataset['y']
y_pred = prediction(eeg_dataset_test.dataset['x'], model_w)


# ## **Confusion Matrix**

# In[ ]:


# perital and ocpital channel
# Note: The following confusion matrix code is a remix of Scikit-Learn's 
# plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
# and Made with ML's introductory notebook - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=6): 
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_Y, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
             horizontalalignment="center",
             color="white" if cm[i, j] > threshold else "black",
             size=text_size)


# In[ ]:


make_confusion_matrix(y_true, y_pred, classes=CLASSES, figsize=(20, 20), text_size=20)


# ### Saving Notebook as a script

# In[ ]:


from IPython import get_ipython
name = 'final.ipynb'
get_ipython().system(f'jupyter nbconvert {name} --to python')


# In[ ]:




