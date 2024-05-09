import argparse
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import DataLoader
import pandas as pd
import os, ssl
from metric import calculate_metrics, SSIM
from datetime import datetime

from dataset_HIDE import HIDEDataset 

from fftformer import fftformer 

if not os.environ.get("PYTHONHTTPSVERIFY", "") and getattr(ssl, "_create_unverified_context", None):
    ssl._create_default_https_context = ssl._create_unverified_context

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint if any")
    parser.add_argument("--checkpoint", type=str, default="checkpoint", help="Checkpoint directory")
    args = parser.parse_args()
    return args

class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, optimizer, epoch, device):
    model.train()
    loss_monitor = AverageMeter()
    criterion = torch.nn.MSELoss()  
    ssim = SSIM(data_range=1.0)
    with tqdm(train_loader) as _tqdm:
        for x, y in _tqdm:  
            x = x.to(device)
            y = y.to(device)

            output = model(x)

            mse_loss = criterion(output, y)
            ssim_loss = torch.clamp((1 - ssim(output, y)), 0, 1)
            loss = mse_loss  
            cur_loss = loss.item()

            sample_num = x.size(0)
            loss_monitor.update(cur_loss, sample_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg), sample_num=sample_num)

    return loss_monitor.avg


def validate(validate_loader, model, epoch, device):
    model.eval()
    loss_monitor = AverageMeter()
    ssim_monitor = AverageMeter()
    pnsr_monitor = AverageMeter()
    calc_ssim = SSIM()
    criterion_mse = torch.nn.MSELoss()
    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for x, y in _tqdm: 
                x = x.to(device)
                y = y.to(device)

                output = model(x)

                mse_loss = criterion_mse(output, y)
                ssim_loss = torch.clamp((1 - calc_ssim(output, y)), 0, 1)
                loss = mse_loss
                cur_loss = loss.item()

                psnr, ssim = calculate_metrics(output, y)

                sample_num = x.size(0)
                loss_monitor.update(cur_loss, sample_num)
                ssim_monitor.update(ssim)
                pnsr_monitor.update(psnr)

                _tqdm.set_postfix(OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg), sample_num=sample_num)

    return loss_monitor.avg, pnsr_monitor.avg, ssim_monitor.avg


def initial_values(validate_loader, device):
    loss_monitor = AverageMeter()
    ssim_monitor = AverageMeter()
    pnsr_monitor = AverageMeter()

    criterion_mse = torch.nn.MSELoss()

    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for x, y in _tqdm:  
                x = x.to(device)
                y = y.to(device)

                loss = criterion_mse(x, y)
                cur_loss = loss.item()
                sample_num = x.size(0)

                psnr, ssim = calculate_metrics(x, y)

                loss_monitor.update(cur_loss, sample_num)
                ssim_monitor.update(ssim)
                pnsr_monitor.update(psnr)

    return loss_monitor.avg, pnsr_monitor.avg, ssim_monitor.avg

def main():
    args = get_args()
    batch_size = 1
    num_workers = 8
    learning_rate = 0.0001
    lr_decay_rate = 0.5
    lr_decay_step = 10
    epochs = 50
    img_size = (1280, 720)  
    training_id = f"{img_size[0]}x{img_size[1]}_fftformer_{datetime.now().strftime('%m-%d-%Y_%H-%M')}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        cudnn.benchmark = True

    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("=> creating model ...")
    model = fftformer().to(device)
    print(f"Model will be trained on {device}")

    train_dataset = HIDEDataset("train", img_size=img_size, limit =50)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"Training dataset loaded with {len(train_dataset)} examples.")

    val_dataset = HIDEDataset("test", img_size=img_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"Validation dataset loaded with {len(val_dataset)} examples.")

    loss, psnr, ssim = initial_values(val_loader, device)
    print(f"Initial validation results - Loss: {loss}, PSNR: {psnr}, SSIM: {ssim}")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_rate)

    training_info = pd.DataFrame()
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train(train_loader, model, optimizer, epoch, device)
        val_loss, val_psnr, val_ssim = validate(val_loader, model, epoch, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"=> [Epoch {epoch}] Best validation loss improved to {best_val_loss:.4f}")
            model_state_dict = model.state_dict()
            save_path = checkpoint_dir.joinpath(f"{training_id}-e{epoch}.pth")
            torch.save({
                "epoch": epoch + 1,
                "state_dict": model_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
            }, str(save_path))

        training_epoch = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": val_loss,
            "valid_SSIM": val_ssim,
            "valid_PSNR": val_psnr,
        }
        training_info = training_info.append(training_epoch, ignore_index=True)

        scheduler.step()

    training_info.to_csv(f"{training_id}.csv", index=False)
    print("=> Training completed")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
