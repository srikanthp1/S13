"""
Implementation of YOLOv3 architecture
"""

import torch
import torch.nn as nn

from pytorch_lightning import LightningModule

import config
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config_dkn53 = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(LightningModule):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(LightningModule):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(LightningModule):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )


class YOLOv3(LightningModule):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()
        # self.learning_rate = 1E-3
        self.learning_rate = 1E-6

        self.scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(config.DEVICE)

        # self.EPOCHS = config.NUM_EPOCHS * 2 // 5
        self.loss_fn = YoloLoss()
        self.scaler = torch.cuda.amp.GradScaler()

        self.train_loader, self.test_loader, self.train_eval_loader = get_loaders(
            train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv"
        )

        # self.loop = tqdm(self.train_loader, leave=True)
        self.losses = []
        self.epoch=0

        self.loss=0

        self.lr_change=[]
        with open('learning_rates.txt', 'r') as f:
            r=[line.rstrip() for line in f]
            for i in r:
                self.lr_change.append(float(i))




    def forward(self, x):
        outputs = []  # for each scale
        route_connections = []
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config_dkn53:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers

    def training_step(self, batch):
        x,y = batch
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = self(x)
            loss = (
                self.loss_fn(out[0], y0, self.scaled_anchors[0])
                + self.loss_fn(out[1], y1, self.scaled_anchors[1])
                + self.loss_fn(out[2], y2, self.scaled_anchors[2])
            )

        self.losses.append(loss.item())
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward(retain_graph=True)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # scheduler.step()

        # update progress bar
        mean_loss = sum(self.losses) / len(self.losses)

        # self.loop.set_postfix(loss=mean_loss)

        return loss

    def on_train_epoch_end(self):
        plot_couple_examples(self, self.test_loader, 0.6, 0.5, self.scaled_anchors)
        # train_fn(train_loader, self, optimizer, loss_fn, scaler, self.scaled_anchors, scheduler)
        # del self.loop
        # self.loop = tqdm(self.train_loader, leave=True)
        mean_loss = sum(self.losses) / len(self.losses)
        self.losses.clear()
        print('lr: ',str(self.optimizer.param_groups[0]['lr']))
        if config.SAVE_MODEL and self.epoch==10:
            save_checkpoint(self, self.optimizer, filename=f"../working/checkpoint_e{self.epoch}.ckpt")
        self.epoch = self.epoch+1
        print(f'loss: {mean_loss}')
        print(f"Currently epoch {self.epoch}")
        if self.epoch==3 or (self.epoch%8==0 and self.epoch%10!=0):
            print("On Train loader:")
            check_class_accuracy(self, self.train_loader, threshold=config.CONF_THRESHOLD)
        # self.optimizer.param_groups[0]['lr'] = self.lr_change[self.epoch] / 1000000

        return self.epoch


    def on_validation_epoch_end(self):
        accuracy = 0
        if self.epoch ==5 or (self.epoch > 0 and self.epoch % 10 == 0):
            check_class_accuracy(self, self.test_loader, threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                self.test_loader,
                self,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
            accuracy = mapval.item()
        return accuracy

    # def configure_optimizers(self):
    #     self.optimizer = torch.optim.Adam(
    #         self.parameters(), lr=self.learning_rate, weight_decay=config.WEIGHT_DECAY
    #     )

    #     self.scheduler=torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=self.learning_rate,epochs=30,steps_per_epoch=518)
    #     lr_scheduler = {'scheduler': self.scheduler, 'interval': 'step'}
        # return {'optimizer': self.optimizer, 'lr_scheduler': lr_scheduler}
        # return self.optimizer

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=config.WEIGHT_DECAY
        )

        return self.optimizer
    ####################
    # DATA RELATED HOOKS
    ####################


    def train_dataloader(self):
        return get_loaders(train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv")[0]
        # return get_dataloader()[0]


    def val_dataloader(self):
        return get_loaders(train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv")[2]
        # return get_dataloader()[1]

    def test_dataloader(self):
        return get_loaders(train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/test.csv")[1]
        # return get_dataloader()[1]





if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")