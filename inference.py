import torch
from config import Config
from model import ExampleModel
from train import validate
from util import build_dataloaders


def inference(config: Config, checkpoint: str):
    test_loader = build_dataloaders(config, mode='test')
    model = ExampleModel(config)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.to(config.device)
    _, acc = validate(config, model, test_loader)
    print(f"Test accuracy: {acc:.3f}")


if __name__ == '__main__':
    config = Config()
    inference(config, './save/epoch9.pth')
