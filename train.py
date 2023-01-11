import torch
import numpy as np
from utils import extract_images, extract_bbox


def train(model, loader, optimizer, device):

    bbox_loss = torch.nn.SmoothL1Loss()
    reg_loss = torch.nn.L1Loss(reduction="sum")
    class_loss = torch.nn.CrossEntropyLoss()

    model.train()
    for i, training_data in enumerate(loader):
        train_images = torch.Tensor(extract_images(training_data)).to(device)
        train_bbox = torch.Tensor(extract_bbox(training_data)).to(device)
        train_label = torch.Tensor(np.ones(train_bbox.shape[0])).to(device)

        N += train_images.shape[0]
        outputs = model(train_images)

        b_loss = 20.0 * bbox_loss(outputs[0], train_bbox)
        c_loss = class_loss(outputs[1], train_label)

        regression_loss += reg_loss(outputs[0], train_bbox).item() / 4.0

        # Backward and optimize
        optimizer.zero_grad()
        b_loss.backward()
        c_loss.backward()
        optimizer.step()
        torch.save(model.state_dict(), "model.pt")
    return regression_loss / N
