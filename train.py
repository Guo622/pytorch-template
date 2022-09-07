import os
import time
import torch
import logging
import numpy as np
from tqdm import tqdm
from config import Config
from model import ExampleModel
from torch.utils.tensorboard import SummaryWriter
from util import build_dataloaders,build_optimizers
from util import setup_logging,setup_seed,set_data_device


def validate(config: Config, model, val_loader):
    model.eval()
    losses = []
    labels = []
    preds = []
    tqdm_loader = tqdm(val_loader)
    tqdm_loader.set_description("validate...")
    with torch.no_grad():
        for batch in tqdm_loader:
            batch = set_data_device(batch, config.device)
            loss, _, pred_label = model(batch)
            losses.append(loss.item())
            labels.extend(batch['label'].cpu().numpy())
            preds.extend(pred_label.cpu().numpy())
    loss = sum(losses)/len(losses)
    acc = sum(np.array(labels) == np.array(preds))/len(labels)
    model.train()
    return loss, acc


def train_and_validate(config:Config, writer:SummaryWriter=None):
    train_loader, val_loader = build_dataloaders(config)
    max_steps = len(train_loader)*config.max_epochs
    model = ExampleModel(config)

    if config.checkpoint is not None:
        model.load_state_dict(torch.load(config.checkpoint, map_location='cpu'), strict=False)

    model.to(config.device)
    optimizer, scheduler = build_optimizers(config, model, max_steps)

    step = 0
    start_time = time.time()
    for epoch in range(config.max_epochs):
        tqdm_loader = tqdm(train_loader)
        for i, batch in enumerate(tqdm_loader):
            model.train()
            batch = set_data_device(batch, config.device)
            loss, acc, _ = model(batch)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
                
            step += 1
            if step % config.print_steps == 0:
                time_per_step = (time.time() - start_time) / step
                remaining_time = time_per_step * (max_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {acc:.3f}")
            tqdm_loader.set_description(f"epoch {epoch} bacth {i} loss:{loss:.3f} acc:{acc:.3f}")
            if writer is not None:
                writer.add_scalar("train_loss", loss.item(), epoch*len(train_loader)+i)
                writer.add_scalar("train_acc", acc, epoch*len(train_loader)+i)

        loss, acc = validate(config, model, val_loader)
        logging.info(f"Epoch {epoch} valid: loss {loss:.3f}, accuracy {acc:.3f}\n")
        print(f"valid: loss {loss:.3f}, accuracy {acc:.3f}\n")
        
        torch.save(model.state_dict(), f"{config.save_path}/epoch{epoch}_{acc:.3f}.pth")


if __name__ == '__main__':

    config = Config()
    
    # config.load_from_dict(f'./config/config.json') # 复现用

    os.makedirs(config.save_path, exist_ok=True)
    os.makedirs(config.log_path, exist_ok=True)

    os.makedirs(config.tblog_path, exist_ok=True)

    setup_seed(config.seed)
    setup_logging(config.log_path)

    logging.info("Args: %s", config.__dict__)
    writer = SummaryWriter(config.tblog_patj)

    train_and_validate(config, writer=writer)
    writer.close()
    config.save_dict(f"./config/config.json")
    
