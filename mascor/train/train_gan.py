import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
from mascor.models.gan import *
from mascor.utils.gan_data_loader import *
from torch.utils.data import DataLoader
from torch.autograd import grad as torch_grad
import os
import pickle
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--target-country', type = str, default = "France", help = "target country")
parser.add_argument("--region", type=str, default="Dunkirk", help="target region")
parser.add_argument('--data-type', type=str, default="wind")
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument("--d-step",type=int, default=2, help="Additional steps for generator training",)
parser.add_argument("--epoch", type=int, default=10000, help="Epoch",)
parser.add_argument("--save-epoch", type=int, default=10, help="Epoch",)
parser.add_argument("--batch-size", type=int, default=512, help="Batch size",)
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate",)
parser.add_argument("--gamma", type=float, default=100, help="weight of MMD loss term",)
parser.add_argument("--MMD-update", action="store_true", help="MMD loss update",)
parser.add_argument("--gp-weight", type=float, default=20.0, help="env config: grid penalty",)
parser.add_argument("--pre-train", action="store_true", help="re-load pre-trained check point",)
parser.add_argument("--pre-train-epoch", type=int, default=0, help="pre-train epoch",)

def MMD(x, y, kernel, device):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  
    dyy = ry.t() + ry - 2. * yy  
    dxy = rx.t() + ry - 2. * zz  

    XX, YY, XY = (torch.zeros(xx.shape, device=device),
                  torch.zeros(xx.shape, device=device),
                  torch.zeros(xx.shape, device=device))

    if kernel == "multiscale":
        bandwidth_range = torch.tensor([0.2, 0.5, 0.9, 1.3], device=device, dtype=torch.float32)
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1
    if kernel == "rbf":
        bandwidth_range = torch.tensor([10, 15, 20, 50], device=device, dtype=torch.float32)
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

def weights_init(m):
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, -0.02, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
# %%
if __name__ == "__main__":
    args = parser.parse_args()
    ROOT = Path(__file__).resolve().parents[2]   
    DATASET_DIR = str(ROOT / "dataset")
    # Loading and instantiate dataloader
    dataset = Dataset(args.target_country, args.region, uni_seq=24, max_seq=24 * 24, data_type=args.data_type,
                      flag='train')
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=True, drop_last=True)  # 1200 by 1 by 24 by 24 should be feded
    # Use GPU if available.
    device = torch.device(args.device if (torch.cuda.is_available()) else "cpu")
    print(device, " will be used.\n")

    # Loading the netG and netD
    if "ele" in args.data_type:
        ch_dim = 2
    else:
        ch_dim = 1
    if args.pre_train:
        print("*-" * 100)
        print("Loading pre-trained checkpoint...")
        save_path = os.path.join(DATASET_DIR, '{country}/{region}/checkpoint_gan/{data_type}_{gp}'.format(
            country=args.target_country,
            region=args.region,
            data_type=args.data_type,
            gp=args.gp_weight))
        checkpoint_path = os.path.join(save_path, 'model_epoch_{epoch}'.format(epoch=5000))
        state_dict = torch.load(checkpoint_path, map_location=args.device)
        netG = generator_1dcnn_24_v2(ch_dim=ch_dim, nz=205).to(args.device)
        netG.load_state_dict(state_dict['netG'])
        netG.train()
        print(netG)
        discriminator = discriminator_1dcnn_24_v2(ch_dim=ch_dim).to(args.device)
        discriminator.load_state_dict(state_dict['discriminator'])
        discriminator.train()
        print(discriminator)
    else:
        netG = generator_1dcnn_24_v2(ch_dim=ch_dim, nz=205).to(device)
        netG.apply(weights_init)
        netG.train()
        print(netG)
        discriminator = discriminator_1dcnn_24_v2(ch_dim=ch_dim).to(device)
        discriminator.apply(weights_init)
        discriminator.train()
        print(discriminator)

    z = torch.randn(20, 205, 1, 1, device=device)
    fixed_noise = z.view(-1, 205)  # for check the training progress

    # Training loop
    start_time = time.time()
    iters = 0

    optimD = optim.RMSprop(discriminator.parameters(), lr=args.lr)
    optimG = optim.RMSprop(netG.parameters(), lr=args.lr)

    discriminator.train()
    netG.train()

    gp_weight = args.gp_weight
    start_epoch = args.pre_train_epoch
    G_losses = []
    D_losses = []
    WD_list = []
    MMD_list = []

    # Preparing for MMD calculation
    if "ele" in args.data_type:
        for i, data in enumerate(data_loader):
            if i == 0:
                weather_train_total = data[0]
                price_train_total = data[1][:, :, 0]
            else:
                weather_train_total = torch.cat((weather_train_total, data[0]), axis=0)
                price_train_total = torch.cat((price_train_total, data[1][:, :, 0]), axis=0)
    else:
        for i, data in enumerate(data_loader):
            if i == 0:
                weather_train_total = data
            else:
                weather_train_total = torch.cat((weather_train_total, data), axis=0)

    for epoch in range(args.epoch):
        epoch_start_time = time.time()

        for i, data in enumerate(data_loader):
            # Get batch siz
            discriminator.train()
            netG.eval()

            if "ele" in args.data_type:
                b_size = data[0].size(0)
            else:
                b_size = data.size(0)

            if "ele" in args.data_type:
                weather, price = data
                data = torch.cat((weather.unsqueeze(-1), price), axis=-1)
                data = data.permute(0, 2, 1)
                real_data = data.reshape(-1, 2, 24, 24).to(device).float()
            else:
                real_data = data.reshape(-1, 1, 24, 24).to(device).float()

            optimD.zero_grad()

            # Real data
            probs_real = discriminator(real_data)  # return 128, 1024,1,1 shape tensor
            errD_real = torch.mean(probs_real)
            D_x = probs_real.mean().item()

            # Fake data
            noise = torch.randn(b_size, 205, 1, 1, device=device)  # shape = (128, 62,1,1)
            noise = noise.view(-1, 205)
            fake_data = netG(noise)
            probs_fake = discriminator(fake_data)
            errD_fake = torch.mean(probs_fake)

            # Gradient penalty calcuation
            alpha = torch.rand(b_size, 1, 1, 1)
            alpha = alpha.expand_as(fake_data)
            alpha = alpha.to(device)
            interpolated = alpha * real_data + (1 - alpha) * fake_data
            interpolated.requires_grad_(True)

            probs_interpolated = discriminator(interpolated)

            gradients = torch_grad(outputs=probs_interpolated, inputs=interpolated,
                                   grad_outputs=torch.ones(probs_interpolated.size()).to(device), create_graph=True,
                                   retain_graph=True)[0]

            gradients = gradients.view(b_size, -1)

            gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

            gradient_penalty = gp_weight * ((gradients_norm - 1) ** 2).mean()

            errD = errD_fake - errD_real + gradient_penalty
            WD = errD_fake - errD_real
            errD.backward()
            optimD.step()

            if i % args.d_step == 0:
                # Updating Generator and QHead
                netG.train()
                discriminator.eval()
                optimG.zero_grad()
                fake_data = netG(noise)
                probs_fake = discriminator(fake_data)
                errG1 = -torch.mean(probs_fake)
                D_G_z2 = probs_fake.mean().item()

                if args.MMD_update:
                    fake_data = netG(noise)
                    probs_fake = discriminator(fake_data)
                    errG1 = -torch.mean(probs_fake)
                    D_G_z2 = probs_fake.mean().item()
                    real_data = real_data.view(b_size, -1)
                    fake_data = fake_data.view(b_size, -1)
                    errMMD = MMD(real_data, fake_data, kernel='rbf', device=device)
                    errG = errG1 + (errMMD) * args.gamma
                    errG.backward()
                    optimG.step()

            # Check progress of training.
            if i == len(data_loader) - 1:
                print(f"[{epoch + 1 + start_epoch}/{args.epoch + start_epoch}]"
                      f"[{i}/{len(data_loader)}]\t"
                      f"Loss_D: {errD.item():.4f}\t"
                      f"Loss_G: {errG.item():.4f}\t"
                      f"Loss_MMDX{args.gamma}: {errMMD.item()*args.gamma:.4f}")

            # Save the losses for plotting.
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            WD_list.append(WD.item())
            MMD_list.append(errMMD.item())
            iters += 1

        epoch_time = time.time() - epoch_start_time
        # Save network weights.
        if (epoch + 1) % args.save_epoch == 0:
            save_path = os.path.join(DATASET_DIR, '{country}/{region}/checkpoint_gan/{data_type}_{gp}'.format(
                country=args.target_country,
                region=args.region, data_type=args.data_type, gp=args.gp_weight))
            if os.path.isdir(save_path) == False:
                os.makedirs(save_path)
            torch.save({
                'netG': netG.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimD': optimD.state_dict(),
                'optimG': optimG.state_dict(),
                'mmd_mode': args.MMD_update,
                'gamma': args.gamma,
            }, os.path.join(save_path, 'model_mmd_{mmd}_epoch_{epoch}'.format(mmd = args.MMD_update,epoch=epoch + 1 + start_epoch)))

    training_time = time.time() - start_time
    print("-" * 50)
    print('Training finished!\nTotal Time for Training: %.2fm' % (training_time / 60))
    print("-" * 50)

    # Save the loss
    losses = {}
    losses['G_loss'] = G_losses
    losses['D_loss'] = D_losses
    losses['WD'] = WD_list
    losses['MMD'] = MMD_list

    with open(os.path.join(save_path, 'losses.pkl'), 'wb') as f:
        pickle.dump(losses, f)