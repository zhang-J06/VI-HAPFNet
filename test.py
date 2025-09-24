import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from data_utils.data_utils import *
# from models.VIFnet_ori import *
from models.VIFnet_msab import *
# from models.VIFnet_dsfe import *
# from models.VIFnet_msab_ir import *
from metrics import ssim, psnr
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from models.mssim import MSSSIM

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='rgbt', help='dataset')
parser.add_argument('--test_imgs', type=str, default='test_imgs', help='Test imgs folder')
opt = parser.parse_args()
dataset = opt.task

# 输出目录
output_dir = f'pred_imgs_{dataset}/'
print("pred_dir:", output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# 模型路径
model_dir = f'trained_models/nh_msab.pk.best.best'

# 设备选择
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载模型
ckp = torch.load(model_dir, map_location=device)
net = vifnet(3, 3)
net = nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.eval()
net = net.to(device)

# 加载测试数据集
loaders_ = {
    'RGB_train': RGB_train_loader,
    'RGB_test': RGB_test_loader,
    'RGB_val': RGB_val_loader
}
loader_test = loaders_['RGB_test']

# 初始化指标累加器
total_ssim = 0
total_psnr = 0
total_samples = 0

# 遍历整个测试数据集
for i, (rgb, ir, gt) in enumerate(loader_test):
    rgb = rgb.to(device)
    ir = ir.to(device)
    gt = gt.to(device)

    with torch.no_grad():
        out, _, _, _, _  = net(rgb, ir)
        batch_ssim = ssim(out, gt).item()
        batch_psnr = psnr(out, gt)
        # print(batch_ssim, batch_psnr)

        # 累加指标
        total_ssim += batch_ssim * rgb.size(0)
        total_psnr += batch_psnr * rgb.size(0)
        total_samples += rgb.size(0)

        # 保存预测结果
        for j in range(rgb.size(0)):
            ts = torch.squeeze(out[j].clamp(0, 1).cpu())
            vutils.save_image(ts, os.path.join(output_dir, f'pred_{i * loader_test.batch_size + j}.png'))

# 计算平均指标
average_ssim = total_ssim / total_samples
average_psnr = total_psnr / total_samples

# 输出平均指标
print('Average SSIM:', average_ssim)
print('Average PSNR:', average_psnr)