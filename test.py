from dataset.test_dataset import TestDataset

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import config
from utils import logger,common
import SimpleITK as sitk
import os
import numpy as np
from models.model import UNet
from utils.metrics import DiceAverage
from collections import OrderedDict


def predict_one_img(model, img_dataset, args):
    dataloader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=0, shuffle=False)
    model.eval()
    test_dice = DiceAverage(args.n_labels)
    
    with torch.no_grad():
        for data in tqdm(dataloader,total=len(dataloader)):
            data = torch.tensor(data)
            data = data.to(device)
            output = model(data)
            # output = nn.functional.interpolate(output, scale_factor=(1//args.slice_down_scale,1//args.xy_down_scale,1//args.xy_down_scale), mode='trilinear', align_corners=False) # 空间分辨率恢复到原始size
            img_dataset.update_result(output.detach().cpu())

    pred = img_dataset.recompone_result()
    pred = torch.argmax(pred,dim=1)

    pred_img = common.to_one_hot_3d(pred,args.n_labels)
    test_dice.update(pred_img)
    
    test_dice = OrderedDict({'Dice_liver': test_dice.avg[1]})
    if args.n_labels==3: test_dice.update({'Dice_tumor': test_dice.avg[2]})
    
    pred = np.asarray(pred.numpy(),dtype='uint8')
    if args.postprocess:
        pass # TO DO
    pred = sitk.GetImageFromArray(np.squeeze(pred,axis=0))

    return test_dice, pred

if __name__ == '__main__':
    args = config.args
    device = torch.device('cpu' if args.cpu else 'cuda')
    # model info
    model = UNet(in_channels=1, out_channels=args.n_labels).to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu)  # multi-GPU
    ckpt = torch.load('{}/best_model.pth'.format(args.save))
    model.load_state_dict(ckpt['net'])

    test_log = logger.Test_Logger(args.save,"test_log")
    # data info
    result_save_path = '{}/result'.format(args.save)
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    
    image_dir = os.path.join(args.test_data_path, 'ribfrac-test-images')
    image_path_list = sorted([os.path.join(image_dir, file)
        for file in os.listdir(image_dir)])
    image_id_list = sorted([x.split("-")[0]
        for x in os.listdir(image_dir)])
        

    for image_id, image_path in zip(image_id_list, image_path_list):
        img_dataset = TestDataset(args, image_path)
        test_dice,pred_img = predict_one_img(model, img_dataset, args)
        test_log.update(image_id, test_dice)
        sitk.WriteImage(pred_img, os.path.join(result_save_path, f"{file_idx}_pred.nii.gz"))
