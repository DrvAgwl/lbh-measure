import torch

import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics.functional import accuracy, iou, dice_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from lbh_measure.data import BagDataset
from lbh_measure.model import DGCNN_semseg
from lbh_measure.utils.util import collate_fn_pad


class ModelBuilder(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = DGCNN_semseg(config)


    def train_dataloader(self):
        train_loader = DataLoader(
            BagDataset(self.config.pcd_dir, self.config.train_data,
                       set_normals=False, downsample_factor=self.config.get('downsample_factor', 0)),
            num_workers=self.config.num_workers,
            pin_memory=self.config.get('pin_memory', False),
            batch_size=self.config.batch_size,
            drop_last=self.config.get('drop_last', False),
            collate_fn=collate_fn_pad
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            BagDataset(self.config.pcd_dir, self.config.test_data,
                       set_normals=False, downsample_factor=self.config.get('downsample_factor', 0)),
            num_workers=self.config.num_workers,
            pin_memory=self.config.get('pin_memory', False),
            batch_size=self.config.test_batch_size,
            drop_last=self.config.get('drop_last', False),
            collate_fn=collate_fn_pad
        )

        return val_loader

    def training_step(self, batch, batch_idx):
        input_tensor, seg = batch[0], batch[1]
        input_tensor = input_tensor.permute(0, 2, 1)

        seg_pred = self.forward(input_tensor)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        pred = seg_pred.softmax(1)  # .max(dim=2)[1]
        loss = self.compute_loss(seg_pred.view(-1, 2), seg.view(-1, 1).squeeze())

        pred = pred.argmax(-1)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=self.config.batch_size)
        output = {"loss": loss, "pred": pred.detach(), "seg": seg.detach()}
        return output

    def validation_step(self, batch, batch_idx):
        input_tensor, seg = batch[0], batch[1]
        input_tensor = input_tensor.permute(0, 2, 1)

        seg_pred = self.forward(input_tensor)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        pred = seg_pred.softmax(1)  # .max(dim=2)[1]
        loss = self.compute_loss(seg_pred.view(-1, 2), seg.view(-1, 1).squeeze())

        pred = pred.argmax(-1)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=self.config.test_batch_size)
        output = {"loss": loss, "pred": pred.detach(), "seg": seg.detach()}
        return output

    def validation_epoch_end(self, outputs):
        pred = []
        gt_class = []
        for index, i in enumerate(outputs):
            pred.append(i["pred"].squeeze(0))
            gt_class.append(i["seg"].squeeze(0))

        pred = torch.nn.utils.rnn.pad_sequence(pred, batch_first=True, padding_value=0)
        seg = torch.nn.utils.rnn.pad_sequence(gt_class, batch_first=True, padding_value=0)

        acc = accuracy(pred, seg)
        calculated_iou = iou(pred, seg, num_classes=2)
        calculated_dice = dice_score(pred, seg)
        self.log("validation_accuracy", acc, on_epoch=True, prog_bar=True)
        self.log("val_iou", calculated_iou, on_epoch=True, prog_bar=True)
        self.log("val_dice", calculated_dice, on_epoch=True, prog_bar=True)

    def training_epoch_end(self, outputs):
        pred = []
        gt_class = []
        for index, i in enumerate(outputs):
            for a, b in zip(i['pred'], i['seg']):
                pred.append(a)
                gt_class.append(b)

        pred = torch.nn.utils.rnn.pad_sequence(pred, batch_first=True, padding_value=0)
        seg = torch.nn.utils.rnn.pad_sequence(gt_class, batch_first=True, padding_value=0)

        acc = accuracy(pred, seg)
        calculated_iou = iou(pred, seg, num_classes=2)
        calculated_dice = dice_score(pred, seg)

        self.log("train_accuracy", acc, on_epoch=True, prog_bar=True)
        self.log("train_iou", calculated_iou, on_epoch=True, prog_bar=True)
        self.log("train_dice", calculated_dice, on_epoch=True, prog_bar=True)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def compute_loss(self, pred, gold, smoothing=True):
        """
        Calculate cross entropy loss, apply label smoothing if needed.
        :param pred:
        :param gold:
        :param smoothing:
        :return:
        """

        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, reduction="mean")

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr, weight_decay=1e-4)
        dis_sch = CosineAnnealingLR(optimizer, self.config.epochs, eta_min=1e-3)
        return [optimizer], [dis_sch]
