import math
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch import nn

from .config import (
    DEVICE,
    DYNAMIC_MODEL_FILENAME,
    MODEL_DIR,
    MODEL_REPO_ID,
    SEQUENCE_LENGTH,
    STATIC_MODEL_FILENAME,
)


# -------------------------------------------------
# Static model: extracts 512 features from each face
# -------------------------------------------------

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels,
        out_channels,
        i_downsample=None,
        stride=1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.batch_norm1 = nn.BatchNorm2d(
            out_channels,
            eps=0.001,
            momentum=0.99,
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding="same",
            bias=False,
        )
        self.batch_norm2 = nn.BatchNorm2d(
            out_channels,
            eps=0.001,
            momentum=0.99,
        )

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.batch_norm3 = nn.BatchNorm2d(
            out_channels * self.expansion,
            eps=0.001,
            momentum=0.99,
        )

        self.i_downsample = i_downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        x = self.relu(
            self.batch_norm1(self.conv1(x))
        )
        x = self.relu(
            self.batch_norm2(self.conv2(x))
        )
        x = self.batch_norm3(self.conv3(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x = x + identity

        return self.relu(x)


class Conv2dSame(nn.Conv2d):
    @staticmethod
    def calculate_padding(
        input_size,
        kernel_size,
        stride,
        dilation,
    ):
        return max(
            (math.ceil(input_size / stride) - 1) * stride
            + (kernel_size - 1) * dilation
            + 1
            - input_size,
            0,
        )

    def forward(self, x):
        height, width = x.size()[-2:]

        pad_height = self.calculate_padding(
            height,
            self.kernel_size[0],
            self.stride[0],
            self.dilation[0],
        )
        pad_width = self.calculate_padding(
            width,
            self.kernel_size[1],
            self.stride[1],
            self.dilation[1],
        )

        if pad_height > 0 or pad_width > 0:
            x = F.pad(
                x,
                [
                    pad_width // 2,
                    pad_width - pad_width // 2,
                    pad_height // 2,
                    pad_height - pad_height // 2,
                ],
            )

        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layer_list,
        num_classes,
        num_channels=3,
    ):
        super().__init__()

        self.in_channels = 64

        self.conv_layer_s2_same = Conv2dSame(
            num_channels,
            64,
            kernel_size=7,
            stride=2,
            groups=1,
            bias=False,
        )

        self.batch_norm1 = nn.BatchNorm2d(
            64,
            eps=0.001,
            momentum=0.99,
        )

        self.relu = nn.ReLU()

        self.max_pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
        )

        self.layer1 = self._make_layer(
            block,
            layer_list[0],
            planes=64,
            stride=1,
        )
        self.layer2 = self._make_layer(
            block,
            layer_list[1],
            planes=128,
            stride=2,
        )
        self.layer3 = self._make_layer(
            block,
            layer_list[2],
            planes=256,
            stride=2,
        )
        self.layer4 = self._make_layer(
            block,
            layer_list[3],
            planes=512,
            stride=2,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(
            512 * block.expansion,
            512,
        )

        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def _make_layer(
        self,
        block,
        blocks,
        planes,
        stride=1,
    ):
        downsample = None
        layers = []

        if (
            stride != 1
            or self.in_channels
            != planes * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    padding=0,
                ),
                nn.BatchNorm2d(
                    planes * block.expansion,
                    eps=0.001,
                    momentum=0.99,
                ),
            )

        layers.append(
            block(
                self.in_channels,
                planes,
                i_downsample=downsample,
                stride=stride,
            )
        )

        self.in_channels = planes * block.expansion

        for _ in range(blocks - 1):
            layers.append(
                block(
                    self.in_channels,
                    planes,
                )
            )

        return nn.Sequential(*layers)

    def extract_features(self, x):
        x = self.relu(
            self.batch_norm1(
                self.conv_layer_s2_same(x)
            )
        )

        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.relu1(x)
        x = self.fc2(x)

        return x


def create_static_model():
    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        num_classes=7,
        num_channels=3,
    )


# -------------------------------------------------
# Dynamic model: analyzes a sequence of face frames
# -------------------------------------------------

class LSTMPyTorch(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=512,
            hidden_size=512,
            batch_first=True,
            bidirectional=False,
        )

        self.lstm2 = nn.LSTM(
            input_size=512,
            hidden_size=256,
            batch_first=True,
            bidirectional=False,
        )

        self.fc = nn.Linear(256, 7)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = self.fc(x[:, -1, :])

        return self.softmax(x)


# -------------------------------------------------
# Download and load the pretrained model weights
# -------------------------------------------------

def download_model(filename):
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_DIR / filename

    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    print(f"Downloading model: {filename}")

    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=filename,
        local_dir=str(MODEL_DIR),
    )

    return Path(downloaded_path)


def load_models():
    device = torch.device(DEVICE)

    static_model_path = download_model(
        STATIC_MODEL_FILENAME
    )
    dynamic_model_path = download_model(
        DYNAMIC_MODEL_FILENAME
    )

    static_model = create_static_model()

    static_weights = torch.load(
        static_model_path,
        map_location=device,
        weights_only=True,
    )

    static_model.load_state_dict(
        static_weights,
        strict=True,
    )

    static_model.to(device)
    static_model.eval()

    dynamic_model = LSTMPyTorch()

    dynamic_weights = torch.load(
        dynamic_model_path,
        map_location=device,
        weights_only=True,
    )

    dynamic_model.load_state_dict(
        dynamic_weights,
        strict=True,
    )

    dynamic_model.to(device)
    dynamic_model.eval()

    return static_model, dynamic_model, device


# -------------------------------------------------
# Simple test
# -------------------------------------------------

def main():
    static_model, dynamic_model, device = load_models()

    with torch.inference_mode():
        test_image = torch.zeros(
            1,
            3,
            224,
            224,
            device=device,
        )

        features = torch.relu(
            static_model.extract_features(test_image)
        )

        sequence = features.unsqueeze(1).repeat(
            1,
            SEQUENCE_LENGTH,
            1,
        )

        result = dynamic_model(sequence)

    print("\nModels loaded successfully")
    print("Device:", device)
    print("Features shape:", tuple(features.shape))
    print("Sequence shape:", tuple(sequence.shape))
    print("Result shape:", tuple(result.shape))


if __name__ == "__main__":
    main()