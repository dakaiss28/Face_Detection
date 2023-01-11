import torch
import torch.torchvision
import torch.nn as nn


class DetectionModel(nn.Module):
    def __init__(self, num_channels: int, num_classes: int, num_box: int):
        super(DetectionModel, self).__init__()

        # We output four numbers. The semantics of these numbers
        # depend on the training set where we order the cy, cy, width, height
        # see data.py : targets_to_tensor
        self.head_bbox = nn.Sequential(
            nn.Conv2d(
                num_channels, 1024, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Dropout(),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout(),
            nn.Conv2d(512, 4 * num_box, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        # We output the logits for all the classes. There is no "no-object class"

        self.head_class = nn.Sequential(
            nn.Conv2d(
                num_channels, 1024, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Dropout(),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Dropout(),
            nn.Conv2d(
                512,
                num_box * (num_classes + 1),
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
        )

    def forward(self, features):

        y_bbox = self.head_bbox(features)
        y_class = self.head_class(features)

        # bbox_regression, class_logits, objectness_probability
        return y_bbox, y_class[:, :-1, :, :], y_class[:, -1, :, :]


# extract features of the resnet
class FeatureExtractor(nn.Module):
    def __init__(self, model_name: str):
        super(FeatureExtractor, self).__init__()
        model = torchvision.models.resnet18(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False
        self.body = nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        return self.body(x)
