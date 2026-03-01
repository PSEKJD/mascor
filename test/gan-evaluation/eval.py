import pickle
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from .utility import *
matplotlib.use('Agg')  # Use Agg backend
import matplotlib.animation as animation
from mascor.models.gan import *
from mascor.utils.gan_data_loader import *
from torch.utils.data import DataLoader
import os
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--target-country', type = str, default = 'France', help= "target country")
parser.add_argument("--region", type=str, default="Dunkirk", help="target region")
parser.add_argument('--data-type', type = str, default = 'wind')
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument("--gp-weight", type=float, default=20.0, help="env config: grid penalty",)
parser.add_argument("--epoch", type=int, default=5000, help="Epoch",)
parser.add_argument("--dis-epoch", type=int, default=100, help="Training epoch of GRU based discriminator",)
parser.add_argument("--perplexity", type=float, default=100, help="Perplexity of t-SNE figure",)

# MMD score
def MMD(x, y, kernel, device):
    # x = x.to(device)
    # y = y.to(device)
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape, device=device),
                  torch.zeros(xx.shape, device=device),
                  torch.zeros(xx.shape, device=device))

    if kernel == "multiscale":

        bandwidth_range = torch.tensor([0.2, 0.5, 0.9, 1.3], device=device).float()
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1
    if kernel == "rbf":
        bandwidth_range = torch.tensor([10, 15, 20, 50], device=device).float()
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)
    return torch.mean(XX + YY - 2. * XY)

def feature_extractor(profile):
    mean_pw = torch.mean(profile, axis=1).view(-1, 1)
    dlt = torch.abs(profile[:, 1:] - profile[:, :-1])
    mean_dlt = torch.mean(dlt, axis=1).view(-1, 1)
    max_dlt = torch.max(dlt, axis=1)[0].view(-1, 1)
    return torch.cat((mean_pw, max_dlt, mean_dlt), dim=1)


def feature_calculation(dataset):
    daily_dataset = dataset  # .reshape(-1, 365, 24)
    mean_power = np.mean(daily_dataset, axis=-1)
    delt = np.abs(daily_dataset[:, 1:] - daily_dataset[:, :-1])
    mean_delt = np.mean(delt, axis=-1)
    max_delt = np.max(delt, axis=-1)
    return np.array([mean_power, mean_delt, max_delt])


def MinMaxScaler(data):
    numerator = data - np.min(data)
    denominator = np.max(data) - np.min(data)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


def dataset_to_array(dataset, args):
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=True, drop_last=True)  # 1200 by 1 by 24 by 24 should be feded
    if "ele" in args.data_type:
        for i, data in enumerate(data_loader):
            if i == 0:
                weather_array = data[0]
                price_array = data[1][:, :, 0]
            else:
                weather_array = torch.cat((weather_array, data[0]), axis=0)
                price_array = torch.cat((price_array, data[1][:, :, 0]), axis=0)
        return weather_array, price_array
    else:
        for i, data in enumerate(data_loader):
            if i == 0:
                weather_array = data
            else:
                weather_array = torch.cat((weather_array, data), axis=0)
        return weather_array


# %%
if __name__ == "__main__":
    args = parser.parse_args()
    args.batch_size = 64
    train_dataset = Dataset(args.target_country, args.region, uni_seq=24, max_seq=24 * 24, data_type=args.data_type,
                            flag='train')
    test_dataset = Dataset(args.target_country, args.region, uni_seq=24, max_seq=24 * 24, data_type=args.data_type,
                           flag='test')
    train_dataset = dataset_to_array(train_dataset, args).numpy()  # batch by 576
    test_dataset = dataset_to_array(test_dataset, args).numpy()  # batch by 576

    # Use GPU if available.
    device = torch.device(args.device if (torch.cuda.is_available()) else "cpu")
    print(device, " will be used.\n")

    # Loading the netG and netD
    if "ele" in args.data_type:
        ch_dim = 2
    else:
        ch_dim = 1
    netG = generator_1dcnn_24_v2(ch_dim=ch_dim, nz=205).to(device)
    REPO_ROOT = Path(__file__).resolve().parents[2]
    CHEKPOINT_DIR = str(REPO_ROOT / "dataset")
    save_path = os.path.join(CHEKPOINT_DIR,
                             '{country}/{region}/checkpoint_gan/{data_type}_{gp}'.format(country=args.target_country,
                                                                                         region=args.region,
                                                                                         data_type=args.data_type,
                                                                                         gp=args.gp_weight))
    epoch_list = [5000, 7000, 9000, 11000, 13000, 15000]
    for target_epoch in epoch_list:
        # Loading check point
        checkpoint_path = os.path.join(save_path, 'model_mmd_True_epoch_{epoch}'.format(epoch=target_epoch))
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            print(f"Checkpoint loaded successfully from {checkpoint_path}")
            netG.load_state_dict(checkpoint['netG'])
            netG.eval()

            # Calculate discriminative score
            z = torch.randn(train_dataset.shape[0] + test_dataset.shape[0], 205, 1, 1, device=args.device)
            z = z.view(-1, 205)
            fake_data = netG(z).detach().cpu()
            fake_data = fake_data.numpy()
            fake_data = fake_data.reshape(-1, 1, 24 * 24)
            fake_data = fake_data.reshape(-1, 24 * 24)
            fake_train_dl = fake_data[:train_dataset.shape[0]]
            fake_test_dl = fake_data[train_dataset.shape[0]:]
            del fake_data

            dim = 1
            hidden_dim = 1
            batch_size = 32
            train_dl = create_dl(fake_train_dl, train_dataset, hidden_dim, batch_size, dim)
            test_dl = create_dl(fake_test_dl, test_dataset, hidden_dim, batch_size, dim)
            test_acc, best_acc = discriminative_score(train_dl, test_dl, args.dis_epoch, device, hidden_dim, dim)
            print('Test accuracy & Train accuray:', test_acc, best_acc)

            # Calculate the MMD of features
            train_dataset_torch = torch.tensor(train_dataset).to(args.device).float()
            test_dataset_torch = torch.tensor(test_dataset).to(args.device).float()
            fake_data = netG(z)
            fake_data = fake_data.view(-1, 24 * 24)
            train_feature = feature_extractor(train_dataset_torch)
            test_feature = feature_extractor(test_dataset_torch)
            fake_feature1 = feature_extractor(fake_data[:train_feature.shape[0]])
            fake_feature2 = feature_extractor(fake_data[train_feature.shape[0]:])

            train_MMD = MMD(train_feature, fake_feature1, kernel='rbf', device=args.device)
            test_MMD = MMD(test_feature, fake_feature2, kernel='rbf', device=args.device)
            print('MMD of train & valid dataset: {train} & {valid}'.format(train=train_MMD, valid=test_MMD))
            del fake_data, fake_feature1, fake_feature2

            # Plotting the t-SNE
            n_components = 2
            tsne = TSNE(n_components=n_components, perplexity=args.perplexity)
            z = torch.randn(1000, 205, 1, 1, device=args.device)
            z = z.view(-1, 205)
            fake_data = netG(z).detach().cpu()
            fake_data = fake_data.reshape(-1, 1, 24 * 24)
            fake_data = fake_data.reshape(-1, 24 * 24)
            fake_data = np.float32(fake_data)

            total_data = np.vstack((fake_data, train_dataset[:100], test_dataset[:100]))
            X_embedded = tsne.fit_transform(total_data)
            X_fake = X_embedded[:fake_data.shape[0]]
            X_train = X_embedded[fake_data.shape[0]:fake_data.shape[0] + train_dataset.shape[0]]
            X_test = X_embedded[fake_data.shape[0] + train_dataset.shape[0]:]
            plt.figure(figsize=(8, 5))
            plt.scatter(X_fake[:, 0], X_fake[:, 1], label='fake', color='grey', alpha=0.5)
            plt.scatter(X_train[:, 0], X_train[:, 1], label='train', color='darkred')
            plt.scatter(X_test[:, 0], X_test[:, 1], label='test', color='blue')
            plt.tick_params(axis='x', direction='in', length=5, pad=10, labelsize=10, labelcolor='black', top=True)
            plt.tick_params(axis='y', direction='in', length=5, pad=10, labelsize=10, labelcolor='black')
            plt.yticks(fontsize=15)
            plt.xticks(fontsize=15)
            plt.legend(fontsize=15)
            plt.tight_layout()
            figure_path = os.path.join(CHEKPOINT_DIR,
                                       '{country}/{region}/checkpoint_gan/{data_type}_{gp}/tSNE_plot_epoch_{epoch}_train_accuray_{train_accuracy}_test_accuray_{test_accuracy}.jpg'.format(
                                           country=args.target_country,
                                           region=args.region, data_type=args.data_type,
                                           gp=args.gp_weight, epoch=target_epoch,
                                           train_accuracy=best_acc, test_accuracy=test_acc))
            plt.savefig(figure_path, dpi=300)
            plt.show()
            plt.close()
            del fake_data, total_data, X_embedded, X_fake, X_train, X_test

            # Plotting the feature distribution plot
            real_data = np.vstack((train_dataset, test_dataset))
            real_feature = np.transpose(feature_calculation(real_data))
            z = torch.randn((train_dataset.shape[0] + test_dataset.shape[0]), 205, 1, 1, device=args.device)
            z = z.view(-1, 205)
            fake_data = netG(z).detach().cpu()
            fake_data = fake_data.reshape(-1, 1, 24 * 24)
            fake_data = fake_data.reshape(-1, 24 * 24)
            fake_data = np.float32(fake_data)
            fake_feature = np.transpose(feature_calculation(fake_data))

            plt.figure(figsize=(8, 5))
            # Create DataFrames for real and fake features
            columns = ['Average power', 'Average fluctuation', 'Climbing power']
            real_df = pd.DataFrame(real_feature, columns=columns)
            fake_df = pd.DataFrame(fake_feature, columns=columns)

            # Add a column to identify the type
            real_df['Type'] = 'Real data'
            fake_df['Type'] = 'Synthetic data'

            # Combine the data into a single DataFrame
            combined_df = pd.concat([real_df, fake_df], ignore_index=True)

            # Convert 'Type' column to category type
            combined_df['Type'] = combined_df['Type'].astype('category')

            # Create the pair plot
            palette = {'Real data': 'darkred', 'Synthetic data': 'gray'}

            # Create the pair plot with custom style and colors
            sns.set(style='whitegrid')  # Set the background style

            pair_plot = sns.pairplot(combined_df, hue='Type', palette=palette, plot_kws={'alpha': 0.5}, diag_kind='kde')

            # Adjusting layout
            pair_plot.fig.subplots_adjust(top=0.95)
            pair_plot.fig.suptitle('Pair Plot of renewable power features', fontsize=16)

            # plt.tight_layout()
            figure_path = os.path.join(CHEKPOINT_DIR,
                                       '{country}/{region}/checkpoint_gan/{data_type}_{gp}/feature_distribution_plot_epoch_{epoch}_train_MMD_{train_MMD}_valid_MMD_{valid_MMD}.jpg'.format(
                                           country=args.target_country,
                                           region=args.region, data_type=args.data_type,
                                           gp=args.gp_weight, epoch=target_epoch,
                                           train_MMD=train_MMD, valid_MMD=test_MMD))
            plt.savefig(figure_path, dpi=300)
            plt.show()
            plt.close()

        else:
            print(f"Checkpoint does not exist: {checkpoint_path}")