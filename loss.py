import torch
import torch.nn as nn
import torch.nn.functional as F
import diffdist.functional as diff_dist
import torch.distributed as dist
import numpy as np
from math import exp

def Sobelxy(x):
    kernelx = [[-1, 0, 1],
              [-2,0 , 2],
              [-1, 0, 1]]
    kernely = [[1, 2, 1],
              [0,0 , 0],
              [-1, -2, -1]]
    kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
    kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
    weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
    weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    sobelx=F.conv2d(x, weightx, padding=1)
    sobely=F.conv2d(x, weighty, padding=1)
    return torch.abs(sobelx)+torch.abs(sobely)
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)                            # sigma = 1.5    shape: [11, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)    # unsqueeze()函数,增加维度  .t() 进行了转置 shape: [1, 1, 11, 11]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()   # window shape: [1,1, 11, 11]
    return window
def std(img,  window_size=9):

    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq

    return sigma1

def L_Grad(image_A, image_B, image_fused):
    image_A_Y = image_A[:, :1, :, :]
    image_B_Y = image_B[:, :1, :, :]
    image_fused_Y = image_fused[:, :1, :, :]
    gradient_A = Sobelxy(image_A_Y)
    gradient_B = Sobelxy(image_B_Y)
    gradient_fused = Sobelxy(image_fused_Y)
    gradient_joint = torch.max(gradient_A, gradient_B)
    Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
    return Loss_gradient

def L_Int(image_A, image_B, image_fused,mask):
    mask = torch.where(mask > 0, 1, 0)
    maskw = mask.to(torch.float32)
    wir = torch.mean(maskw)
    image_A_Y = image_A[:, :1, :, :]
    image_B_Y = image_B[:, :1, :, :]
    image_fused_Y = image_fused[:, :1, :, :]
    # x_in_max=torch.add(image_A_Y,image_B_Y)/2
    # x_in_max = torch.max(image_A_Y, image_B_Y)
    int_ir = F.l1_loss(image_A_Y * mask, image_fused_Y)
    int_vi = F.l1_loss(image_B_Y, image_fused_Y) * wir
    # int_ir = F.l1_loss(image_A_Y, image_fused_Y)
    # int_vi = F.l1_loss(image_B_Y, image_fused_Y)
    loss_in =int_ir + int_vi
    return loss_in

# class Sobelxy(nn.Module):
#     def __init__(self, device):
#         super(Sobelxy, self).__init__()
#         kernelx = [[-1, 0, 1],
#                   [-2,0 , 2],
#                   [-1, 0, 1]]
#         kernely = [[1, 2, 1],
#                   [0,0 , 0],
#                   [-1, -2, -1]]
#         # 这里不行就采用expend_dims
#         kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
#         kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
#         self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device=device)
#         self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device=device)
#     def forward(self,x):
#         sobelx=F.conv2d(x, self.weightx, padding=1)
#         sobely=F.conv2d(x, self.weighty, padding=1)
#         return torch.abs(sobelx), torch.abs(sobely)
def Fusion_loss(vi, ir, fu,mask, weights=[10, 10], device='cuda'):

    vi_gray = torch.mean(vi, 1, keepdim=True)
    fu_gray = torch.mean(fu, 1, keepdim=True)
    sobelconv=Sobelxy(device)
    mask = torch.where(mask > 0, 1, 0)
    # 梯度损失
    vi_grad_x, vi_grad_y = sobelconv(vi_gray)
    ir_grad_x, ir_grad_y = sobelconv(ir)
    fu_grad_x, fu_grad_y = sobelconv(fu_gray)
    grad_joint_x = torch.max(vi_grad_x, ir_grad_x)
    grad_joint_y = torch.max(vi_grad_y, ir_grad_y)
    loss_grad = F.l1_loss(grad_joint_x, fu_grad_x) + F.l1_loss(grad_joint_y, fu_grad_y)

    ## 强度损失
    # std_ir = std(ir)
    # std_vi = std(vi)
    # (std_ir - std_vi)
    loss_intensity = torch.mean((torch.pow((fu - vi), 2)))+torch.mean(mask * torch.abs((fu_gray - ir)))

    loss_total = weights[0] * loss_grad + weights[1] * loss_intensity
    return loss_total

def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous() for _ in range(dist.get_world_size())]
    out_list = diff_dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()

def Contrastive_loss(image_x, text_x):
    cross_entropy=nn.CrossEntropyLoss()
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    batch_size = image_x.shape[0]
    # get label globally
    labels = torch.arange(batch_size, dtype=torch.long, device=image_x.device)

    # [B, C]
    image_x = F.normalize(image_x, dim=-1)
    text_x = F.normalize(text_x, dim=-1)
    text_x = text_x.to(torch.float32)
    logits_per_img = image_x @ text_x.t()
    logits_per_text = text_x @ image_x.t()

    logit_scale = torch.clamp(logit_scale.exp(), max=100)
    loss_img = cross_entropy(logits_per_img * logit_scale, labels)
    loss_text = cross_entropy(logits_per_text * logit_scale, labels)

    loss = 0.5 * (loss_img + loss_text)

    return loss
# class Fusionloss(nn.Module):
#     def __init__(self):
#         super(Fusionloss, self).__init__()
#         self.sobelconv=Sobelxy()
#
#     def forward(self,image_vis,image_ir,generate_img):
#         image_y=image_vis[:,:1,:,:]
#         x_in_max=torch.max(image_y,image_ir)
#         loss_in=F.l1_loss(x_in_max,generate_img)
#         y_grad=self.sobelconv(image_y)
#         ir_grad=self.sobelconv(image_ir)
#         generate_img_grad=self.sobelconv(generate_img)
#         x_grad_joint=torch.max(y_grad,ir_grad)
#         loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
#         loss_total=loss_in+10*loss_grad
#         return loss_total,loss_in,loss_grad
#
# class Sobelxy(nn.Module):
#     def __init__(self):
#         super(Sobelxy, self).__init__()
#         kernelx = [[-1, 0, 1],
#                   [-2,0 , 2],
#                   [-1, 0, 1]]
#         kernely = [[1, 2, 1],
#                   [0,0 , 0],
#                   [-1, -2, -1]]
#         kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
#         kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
#         self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
#         self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
#     def forward(self,x):
#         sobelx=F.conv2d(x, self.weightx, padding=1)
#         sobely=F.conv2d(x, self.weighty, padding=1)
#         return torch.abs(sobelx)+torch.abs(sobely)