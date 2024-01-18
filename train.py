"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import config
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import YOLOv3
from fp_model import teacher_YOLOv3
from error_model import error_YOLOv3
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

writer = SummaryWriter("runs/YOLOv3")


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors, epoch):
    counter = 0
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )
        if epoch == 0 and counter == 0:
            plot_loss(loss.item(), epoch)
            counter += 1


        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
    mean_loss = sum(losses) / len(losses)
    loop.set_postfix(loss=mean_loss)
    plot_loss(mean_loss, epoch+51)





def kt_train_fn(train_loader, model, teacher_model, optimizer, loss_fn, scaler, scaled_anchors, epoch):
    loop = tqdm(train_loader, leave=True)
    losses = []
    l2_values_lst = []
    l2 = torch.nn.MSELoss()
    counter = 0
    
    for batch_idx, (x, y) in enumerate(loop):
        # out, teacher_fr = teacher_model(x) ###---> CPU 
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE), 
        )

        with torch.cuda.amp.autocast():
            teacher_out, teacher_fr = teacher_model(x)
            out, student_fr = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )
            
            final_l2_value = 0

        #####################################################
        ################ KNOWLEDGE TRANSFER #################
        #####################################################

            for idx in range(2): # TWO FEATURE RESPONSES 
                teacher_response = teacher_fr[idx].to(config.DEVICE)
                student_response = student_fr[idx].to(config.DEVICE)

                for pic in range(teacher_response.size(0)): # Batch Size
                    for channel_idx in range(teacher_response.size(1)): # Channel Size
                        l2_value = l2(teacher_response[pic][channel_idx][:][:], student_response[pic][channel_idx][:][:])
                        l2_value /= (teacher_response.size(2) * teacher_response.size(2)) # NORMIERUNG AUF PIXEL ANZAHL PASST!
                        final_l2_value += l2_value
                    
                    final_l2_value /= teacher_response.size(1)
                
            final_l2_value = final_l2_value / 32 # Normierung auf Anzahl an Bildern
            
        

        #####################################################
        ################ KNOWLEDGE TRANSFER #################
        #####################################################
            if epoch == 0 and counter == 0: ## VOR DEM TRAINING ERSTEN LOSS PLOTTEN!!!
                print(final_l2_value)
                print(loss)
                loss += final_l2_value
                plot_loss(loss, final_l2_value, epoch)
                counter += 1
            else: 
            
                loss += final_l2_value
        

        l2_values_lst.append(final_l2_value)
        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward() ## !!!
        scaler.step(optimizer)
        scaler.update()

    mean_loss = sum(losses) / len(losses)
    mean_l2 = sum(l2_values_lst) / len(l2_values_lst)
    plot_loss(mean_loss, mean_l2, epoch+1)
        

def main():
    #model = YOLOv3().to(config.DEVICE)
    teacher_model = teacher_YOLOv3().to(config.DEVICE)
    error_model = error_YOLOv3().to(config.DEVICE)
    
    optimizer = optim.Adam(
        error_model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path=config.DATASET + "/train.csv", test_csv_path=config.DATASET + "/1000examples_test.csv"
    )

    if config.LOAD_MODEL:
        #load_checkpoint(
        #    config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE
        #)
        teacher_model.load_state_dict(torch.load('YOLOv3_weights.pt'))
        error_model.load_state_dict(torch.load('YOLOv3_weights.pt'))
        print("Success!")
    
        

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):      
        #plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
        kt_train_fn(train_loader, error_model, teacher_model, optimizer, loss_fn, scaler, scaled_anchors, epoch)
        #train_fn(train_loader, error_model, optimizer, loss_fn, scaler, scaled_anchors, epoch)

        #if config.SAVE_MODEL:
        #    save_checkpoint(model, optimizer, filename=f"checkpoint.pth.tar")

        #print(f"Currently epoch {epoch}")
        #print("On Train Eval loader:")
        #print("On Train loader:")
        #check_class_accuracy(model, train_loader, threshold=config.CONF_THRESHOLD)

        # if epoch > 0s and epoch % 3 == 0:
            # check_class_accuracy(model, test_loader, threshold=config.CONF_THRESHOLD)
        """pred_boxes, true_boxes = get_evaluation_bboxes(
                test_loader,
                error_model,
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
        print(f"MAP: {mapval.item()}")"""
        #teacher_model.train()
   
    import time
    time.sleep(10)

    """checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
           }
    save_checkpoint(checkpoint, filename="binarized_YOLOv3") """

    torch.save(error_model.state_dict(), "kt_error_YOLOv3_weights.pt")

    time.sleep(10)


def plot_loss(loss, l2, epoch):
    writer.add_scalar('mean loss', loss, epoch)
    writer.add_scalar("mean l2 loss", l2, epoch)


if __name__ == "__main__":
    main()