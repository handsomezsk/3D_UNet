from dataset.train_dataset import TrainDataset

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import config
import time

from models.model import UNet

from utils import logger, weights_init, metrics, common, loss
import os
import numpy as np
from collections import OrderedDict


def train(model, train_loader, optimizer, loss_func, n_labels, alpha):
    print("=======Epoch:{}=======lr:{}".format(epoch,optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
        data, target = data.float(), target.squeeze(1).long()
        #data, target = data.float(), target.long()
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

def val(model, val_loader, loss_func, n_labels):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    with torch.no_grad():
        for idx,(data, target) in tqdm(enumerate(val_loader),total=len(val_loader)):
            data, target = data.float(), target.squeeze(1).long()
            #data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss=loss_func(output, target)
            
            val_loss.update(loss.item(),data.size(0))
            val_dice.update(output, target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_liver': val_dice.avg[1]})
    if n_labels==3: val_log.update({'Val_dice_tumor': val_dice.avg[2]})
    return val_log


if __name__ == '__main__':
    args = config.args
    save_path = args.save
    if not os.path.exists(save_path): os.mkdir(save_path)
    device = torch.device('cpu' if args.cpu else 'cuda')
    # data info
    train_image_dir = os.path.join(args.data_path, 'ribfrac-train-images')
    train_label_dir = os.path.join(args.data_path, 'ribfrac-train-labels')
    val_image_dir = os.path.join(args.data_path, 'ribfrac-val-images')
    val_label_dir = os.path.join(args.data_path, 'ribfrac-val-labels')
    train_data = TrainDataset(args, train_image_dir, train_label_dir)
    train_loader = TrainDataset.get_dataloader(args, train_data)
    val_data = TrainDataset(args, val_image_dir, val_label_dir)
    val_loader = TrainDataset.get_dataloader(args, val_data)
    
    # model info
    model = UNet(in_channels=1, out_channels=args.n_labels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.weight is not None:
        checkpoint = torch.load(args.weight)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        log = logger.Train_Logger(save_path,"train_log",init=os.path.join(save_path,"train_log.csv"))
        best = [log.log.idxmax()['Val_dice_liver']+1, log.log.max()['Val_dice_liver']]
    else:
        log = logger.Train_Logger(save_path,"train_log")
        model.apply(weights_init.init_model)
        start_epoch = 1
        best = [0, 0]

    common.print_network(model)
    model = torch.nn.DataParallel(model, device_ids=args.gpu)  # multi-GPU
 
    loss = loss.TverskyLoss()

    trigger = 0
    alpha = 0.4
    for epoch in range(start_epoch, args.epochs + start_epoch ):
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
    ax = log.log.plot(x='epoch', y='Val_dice_liver', grid=True, title='Val_dice_liver')
    fig = ax.get_figure()
    fig.savefig(os.path.join(save_path, 'fig.png'))
    plt.show()