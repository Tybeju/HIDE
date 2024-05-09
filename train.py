import argparse
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import DataLoader
import pandas as pd
from utils.metrics import calculate_metrics
from datetime import datetime
from typing import Tuple
from dataset import HIDEDataset
from torch.nn import Module

from models.fftformer import fftformer as FFTTransformer
from models.attention_unet import AttentionUNet


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint", help="Checkpoint directory"
    )
    parser.add_argument(
        "--approach_id", type=str, default=None, help="Identifier for the approach used"
    )
    parser.add_argument(
        "--model_id", type=str, default="attention_unet", help="Model name or type"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=(128, 128),
        help="Image size as a tuple (width, height)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for processing"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    return parser.parse_args()


class AverageMeter:
    # TODO: might be done smarter
    def __init__(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def train(
    train_loader: DataLoader,
    model: Module,
    optimizer: Optimizer,
    epoch: int,
    device: torch.device,
) -> float:
    model.train()
    loss_monitor = AverageMeter()
    criterion = torch.nn.MSELoss()
    with tqdm(train_loader) as _tqdm:
        for x, y in _tqdm:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)
            # TODO: add other loss functions (eg. SSIM rel. - https://arxiv.org/abs/1812.11941)
            cur_loss = loss.item()

            sample_num = x.size(0)
            loss_monitor.update(cur_loss, sample_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.set_postfix(
                OrderedDict(stage="train", epoch=epoch, loss=loss_monitor.avg),
                sample_num=sample_num,
            )

    return loss_monitor.avg


def validate(
    validate_loader: DataLoader, model: Module, epoch: int, device: torch.device
) -> Tuple[float, float, float]:
    model.eval()
    loss_monitor = AverageMeter()
    ssim_monitor = AverageMeter()
    pnsr_monitor = AverageMeter()
    criterion_mse = torch.nn.MSELoss()
    with torch.no_grad():
        with tqdm(validate_loader) as _tqdm:
            for x, y in _tqdm:
                x = x.to(device)
                y = y.to(device)

                output = model(x)
                loss = criterion_mse(output, y)
                cur_loss = loss.item()

                psnr, ssim = calculate_metrics(output, y)

                sample_num = x.size(0)
                loss_monitor.update(cur_loss, sample_num)
                ssim_monitor.update(ssim, sample_num)
                pnsr_monitor.update(psnr, sample_num)

                _tqdm.set_postfix(
                    OrderedDict(stage="val", epoch=epoch, loss=loss_monitor.avg),
                    sample_num=sample_num,
                )

    return loss_monitor.avg, pnsr_monitor.avg, ssim_monitor.avg


def initial_values(
    validate_loader: DataLoader, device: torch.device
) -> Tuple[float, float, float]:
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
                ssim_monitor.update(ssim, sample_num)
                pnsr_monitor.update(psnr, sample_num)

    return loss_monitor.avg, pnsr_monitor.avg, ssim_monitor.avg


def main() -> None:
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_id: str = (
        f"{args.approach_id}_{datetime.now().strftime('%m-%d-%Y_%H-%M')}"
        if args.approach_id
        else f"{args.img_size[0]}x{args.img_size[1]}_{args.model_id}_{datetime.now().strftime('%m-%d-%Y_%H-%M')}"
    )

    if device.type == "cuda:0":
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    checkpoint_dir: Path = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"=> creating model: {args.model_id}...")
    model: Module = (
        AttentionUNet().to(device)
        if args.model_id == "attention_unet"
        else FFTTransformer().to(device)
    )
    print(f"Model will be trained on {device}")

    # TODO: add to args HIDEDataset params if seems helpful
    train_dataset = HIDEDataset("train", img_size=args.img_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    print(f"Training dataset loaded with {len(train_dataset)} examples.")

    val_dataset = HIDEDataset("test", img_size=args.img_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    print(f"Validation dataset loaded with {len(val_dataset)} examples.")

    optimizer: Optimizer = Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    training_info = pd.DataFrame()
    best_val_loss: float = float("inf")

    for epoch in range(args.epochs):
        train_loss = train(train_loader, model, optimizer, epoch, device)
        val_loss, val_psnr, val_ssim = validate(val_loader, model, epoch, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(
                f"=> [Epoch {epoch}] Best validation loss improved to {best_val_loss:.4f}"
            )
            model_state_dict = model.state_dict()
            save_path = checkpoint_dir.joinpath(f"{training_id}-e{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                str(save_path),
            )

        #  TODO: verify metrics, add new
        training_epoch_df = pd.DataFrame(
            [
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": val_loss,
                    "valid_SSIM": val_ssim,
                    "valid_PSNR": val_psnr,
                }
            ]
        )
        training_info = pd.concat([training_info, training_epoch_df], ignore_index=True)

        scheduler.step()
        training_info.to_csv(f"{training_id}.csv", index=False)

    print("=> Training completed")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
