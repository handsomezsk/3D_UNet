from dataset.train_dataset import TrainDataset

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config

from models.model import UNet

from utils import logger, weights_init, metrics, common, loss
import os
import numpy as np
from collections import OrderedDict

def val(model, val_loader, loss_func, n_labels):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    with torch.no_grad():
        for idx,(data, target) in tqdm(enumerate(val_loader),total=len(val_loader)):
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss=loss_func(output, target)
            
            val_loss.update(loss.item(),data.size(0))
            val_dice.update(output, target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_liver': val_dice.avg[1]})
    if n_labels==3: val_log.update({'Val_dice_tumor': val_dice.avg[2]})
    return val_log

def train(model, train_loader, optimizer, loss_func, n_labels, alpha):
    print("=======Epoch:{}=======lr:{}".format(epoch,optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
        data, target = data.float(), target.squeeze(1).long()
        target = common.to_one_hot_3d(target,n_labels)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = loss_func(output, target)

        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.item(),data.size(0))
        train_dice.update(output, target)

    val_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice_liver': train_dice.avg[1]})
    if n_labels==3: val_log.update({'Train_dice_tumor': train_dice.avg[2]})
    return val_log

if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('outputs', args.save)
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # data info
    train_image_dir = os.path.join(args.train_data_path, 'ribfrac-train-images')
    train_label_dir = os.path.join(args.train_data_path, 'ribfrac-train-labels')
    val_image_dir = os.path.join(args.train_data_path, 'ribfrac-val-images')
    val_label_dir = os.path.join(args.train_data_path, 'ribfrac-val-labels')
    train_data = TrainDataset(args, train_image_dir, train_label_dir)
    train_loader = TrainDataset.get_dataloader(args, train_data)
    val_data = TrainDataset(args, val_image_dir, val_label_dir)
    val_loader = TrainDataset.get_dataloader(args, val_data)

    
    # model info
    model = UNet(in_channels=1, out_channels=args.n_labels).to(device)

    model.apply(weights_init.init_model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    common.print_network(model)
    model = torch.nn.DataParallel(model, device_ids=args.gpu)  # multi-GPU
 
    loss = loss.TverskyLoss()

    log = logger.Train_Logger(save_path,"train_log")

    best = [0,0] # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器
    alpha = 0.4 # 深监督衰减系数初始值
    for epoch in range(1, args.epochs + 1):
        common.adjust_learning_rate(optimizer, epoch, args)
        train_log = train(model, train_loader, optimizer, loss, args.n_labels, alpha)
        val_log = val(model, val_loader, loss, args.n_labels)
        log.update(epoch,train_log,val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['Val_dice_liver'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice_liver']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0],best[1]))

        # 深监督系数衰减
        if epoch % 30 == 0: alpha *= 0.8

        # early stopping
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()    