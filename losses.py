
import torch
import torch.nn as nn

class LIH_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduce=True, size_average=False)
    def forward(self, secret, cover , stego ,rec, steg_low, cover_low,rec_weight,guide_weight,freq_weight):
        N, _, H, W = secret.size()
        out = {}
        guide_loss = self.mse(stego,cover)
        reconstruction_loss = self.mse(rec,secret)
        freq_loss = self.mse(steg_low,cover_low)
        hide_loss = rec_weight*reconstruction_loss  + freq_weight*freq_loss  +guide_weight*guide_loss
        out['g_loss'] = guide_loss
        out['r_loss'] = reconstruction_loss
        out['f_loss'] = freq_loss
        out['hide_loss'] = hide_loss
        return out


class LSR_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduce=True, size_average=False)
    def forward(self,secret_img,cover_img,steg_clean,rec_img):
        N, _, H, W = secret_img.size()
        out = {}
        lossc = self.mse(cover_img,steg_clean)
        losss = self.mse(secret_img,rec_img)
        loss =lossc + 3*losss
        out['loss'] = loss
        return out
 


