
import warnings

from yolo_model.pdestre_dataset import PDestreDataset, evaluate_pdestre

from yolo_model.yolo_pdestre import YoloV3Pdestre
from zoo.pytorch_yolo_v3.utils.logger import Logger

warnings.filterwarnings('ignore', category=UserWarning)

from terminaltables import AsciiTable

from yolo_model.yolov3 import YoloV3
import os
import time
import datetime
import torch
from torch.utils.data import DataLoader

from zoo.pytorch_yolo_v3.test import evaluate, ListDataset

train_path = r'f:\my\Prog\CV\Datasets\coco\trainvalno5k.txt'
validation_path = r'f:\my\Prog\CV\Datasets\coco\5k.txt'
epochs = 10000
evaluation_interval = 5
img_size = 416
checkpoint_interval = 5
gradient_accumulations = 32

def train_pdestre(yolo:YoloV3Pdestre):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    # parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    # parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    # parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    # parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    # parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    # parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    # parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    # parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    # parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    # parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    # parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    # opt = parser.parse_args()
    # print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger("logs")
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)


    # Get dataloader

    dataset = PDestreDataset()
    # dataset = ListDataset(r'f:\my\Prog\CV\Datasets\coco\trainvalno5k.txt')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(yolo.model.parameters(), lr=1e-4)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(epochs):
        yolo.model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):

            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device)
            targets = targets.to(device)
            targets.requires_grad = False

            loss, outputs = yolo.model(imgs, targets)
            loss.backward()

            if batches_done % gradient_accumulations == 0:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(yolo.model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % y.metrics.get(metric, 0) for y in yolo.model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, y in enumerate(yolo.model.yolo_layers):
                    for name, metric in y.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j + 1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f" ETA {time_left}"

            print(log_str)

            yolo.model.seen += imgs.size(0)

        if (epoch + 1) % evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            precision, recall, AP, f1, ap_class = evaluate_pdestre(
                yolo,
                path=None,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=img_size,
                batch_size=8,
            )
            evaluation_metrics = [
                ("val_precision", precision.mean()),
                ("val_recall", recall.mean()),
                ("val_mAP", AP.mean()),
                ("val_f1", f1.mean()),
            ]
            logger.list_of_scalars_summary(evaluation_metrics, epoch)

            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, yolo.class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")

        if epoch % checkpoint_interval == 0:
            torch.save(yolo.model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)


if __name__ == "__main__":
    m = YoloV3Pdestre(weights_path=r'f:\my\Prog\CV\deepvideos\test\models\yolov3_pdestre_8may_loss=3.pth')
    # m = YoloV3()
    train_pdestre(m)