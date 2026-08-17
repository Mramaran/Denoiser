import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basicblock import ResBlock, downsample_strideconv, upsample_convtranspose

class DRUNet(nn.Module):
    def __init__(self, in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, sf=1):
        super(DRUNet, self).__init__()
        
        self.m_head = nn.Conv2d(in_nc, nc[0], 3, 1, 1, bias=True)
        
        # Encoder
        self.m_down1 = nn.Sequential(*[ResBlock(nc[0], nc[0], nc[0]) for _ in range(nb)])
        self.m_down2 = nn.Sequential(*[ResBlock(nc[1], nc[1], nc[1]) for _ in range(nb)])
        self.m_down3 = nn.Sequential(*[ResBlock(nc[2], nc[2], nc[2]) for _ in range(nb)])
        
        self.m_down1_stride = downsample_strideconv(nc[0], nc[1])
        self.m_down2_stride = downsample_strideconv(nc[1], nc[2])
        self.m_down3_stride = downsample_strideconv(nc[2], nc[3])
        
        # Bottleneck
        self.m_body = nn.Sequential(*[ResBlock(nc[3], nc[3], nc[3]) for _ in range(nb)])
        
        # Decoder
        self.m_up3_convtranspose = upsample_convtranspose(nc[3], nc[2])
        self.m_up3_conv1x1 = nn.Conv2d(nc[2]*2, nc[2], 1, 1, 0, bias=True)
        self.m_up3 = nn.Sequential(*[ResBlock(nc[2], nc[2], nc[2]) for _ in range(nb)])
        
        self.m_up2_convtranspose = upsample_convtranspose(nc[2], nc[1])
        self.m_up2_conv1x1 = nn.Conv2d(nc[1]*2, nc[1], 1, 1, 0, bias=True)
        self.m_up2 = nn.Sequential(*[ResBlock(nc[1], nc[1], nc[1]) for _ in range(nb)])
        
        self.m_up1_convtranspose = upsample_convtranspose(nc[1], nc[0])
        self.m_up1_conv1x1 = nn.Conv2d(nc[0]*2, nc[0], 1, 1, 0, bias=True)
        self.m_up1 = nn.Sequential(*[ResBlock(nc[0], nc[0], nc[0]) for _ in range(nb)])
        
        self.sf = sf
        if sf > 1:
            # Extract features at LR, upsample once at the end. Cheaper than
            # pre-upsampling and free of transposed-conv checkerboard artifacts.
            self.m_tail = nn.Sequential(
                nn.Conv2d(nc[0], out_nc * sf * sf, 3, 1, 1, bias=True),
                nn.PixelShuffle(sf),
            )
        else:
            self.m_tail = nn.Conv2d(nc[0], out_nc, 3, 1, 1, bias=True)

    def forward(self, x, sigma=None):
        # x: (B, 1, H, W)
        B, C, H, W = x.shape
        inp = x  # keep the raw LR image for the global skip

        if sigma is not None:
            if isinstance(sigma, (float, int)):
                noise_map = torch.full((B, 1, H, W), float(sigma), dtype=x.dtype, device=x.device)
            elif sigma.dim() == 0:
                noise_map = torch.full((B, 1, H, W), sigma.item(), dtype=x.dtype, device=x.device)
            else:
                noise_map = sigma.view(B, 1, 1, 1).expand(B, 1, H, W)
            x = torch.cat([x, noise_map], dim=1)  # (B, 2, H, W)

        x = self.m_head(x)
        
        x1 = self.m_down1(x)
        x_d1 = self.m_down1_stride(x1)
        
        x2 = self.m_down2(x_d1)
        x_d2 = self.m_down2_stride(x2)
        
        x3 = self.m_down3(x_d2)
        x_d3 = self.m_down3_stride(x3)
        
        x_body = self.m_body(x_d3)
        
        x_u3 = self.m_up3_convtranspose(x_body)
        x_u3 = torch.cat([x_u3, x3], dim=1)
        x_u3 = self.m_up3_conv1x1(x_u3)
        x_u3 = self.m_up3(x_u3)
        
        x_u2 = self.m_up2_convtranspose(x_u3)
        x_u2 = torch.cat([x_u2, x2], dim=1)
        x_u2 = self.m_up2_conv1x1(x_u2)
        x_u2 = self.m_up2(x_u2)
        
        x_u1 = self.m_up1_convtranspose(x_u2)
        x_u1 = torch.cat([x_u1, x1], dim=1)
        x_u1 = self.m_up1_conv1x1(x_u1)
        x_u1 = self.m_up1(x_u1)
        
        out = self.m_tail(x_u1)

        if self.sf > 1:
            # Learn only the residual over bicubic interpolation.
            out = out + F.interpolate(inp, scale_factor=self.sf,
                                      mode='bicubic', align_corners=False)

        return out
