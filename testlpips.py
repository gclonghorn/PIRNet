import argparse
import os
import torch
import lpips
from tqdm import tqdm
import PIL.Image as Image
from typing import Tuple, Union
import numpy as np
from pytorch_msssim import ms_ssim

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str)
parser.add_argument('-d1','--dir1', type=str)
parser.add_argument('-o','--out', type=str)
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

def compute_metrics(
        a: Union[np.array, Image.Image],
        b: Union[np.array, Image.Image],
        max_val: float = 255.0,
) -> Tuple[float, float]:
    """Returns PSNR and MS-SSIM between images `a` and `b`. """
    if isinstance(a, Image.Image):
        a = np.asarray(a)
    if isinstance(b, Image.Image):
        b = np.asarray(b)

    a = torch.from_numpy(a.copy()).float().unsqueeze(0)
    if a.size(3) == 3:
        a = a.permute(0, 3, 1, 2)
    b = torch.from_numpy(b.copy()).float().unsqueeze(0)
    if b.size(3) == 3:
        b = b.permute(0, 3, 1, 2)

    mse = torch.mean((a - b) ** 2).item()
    p = 20 * np.log10(max_val) - 10 * np.log10(mse)
    m = ms_ssim(a, b, data_range=max_val).item()
    return p, m

## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)
if(opt.use_gpu):
	loss_fn.cuda()

# crawl directories
f = open(opt.out,'w')
files = os.listdir(opt.dir0)
dist_=[]
psnr_=[]
ssim_=[]
for file in tqdm(files):
	if(os.path.exists(os.path.join(opt.dir1,file))):
		image0 = Image.open(os.path.join(opt.dir0,file))
		image1 = Image.open(os.path.join(opt.dir1,file))
		# Load images
		img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
		img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir1,file)))

		if(opt.use_gpu):
			img0 = img0.cuda()
			img1 = img1.cuda()

		# Compute distance
		dist01 = loss_fn.forward(img0,img1)
		p,m = compute_metrics(image0,image1)
		dist_.append(dist01.mean().item())
		psnr_.append(p)
		ssim_.append(m)
		f.writelines('%s: %.6f\n'%(file,dist01))
		f.writelines('%s: %.6f\n'%(file,p))
		f.writelines('%s: %.6f\n'%(file,m))
print(len(dist_))
avg = sum(dist_)/len(files)
avg_psnr = sum(psnr_)/len(files)
avg_ssim = sum(ssim_)/len(files)
print('Avarage lpips: %.3f' % avg)
print('Avarage psnr: %.3f' % avg_psnr)
print('Avarage ssim: %.3f' % avg_ssim)
f.write('%.3f\n'%avg)
f.write('%.3f\n'%avg_psnr)
f.write('%.3f\n'%avg_ssim)
f.close()
