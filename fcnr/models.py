import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import Dict, tictoc, quantise
from .layers import *
from .metrics import laplace_cdf, calc_rate
from einops import rearrange
import math

__all__ = ['DebugModel', 'StereoBaseline', 'StereoAttentionModelPlus']

class PositionalEncoding(nn.Module):
    def __init__(self, pe_embed_b, pe_embed_l):
        super(PositionalEncoding, self).__init__()
        if pe_embed_b == 0:
            self.embed_length = 1
            self.pe_embed = False
        else:
            self.lbase = float(pe_embed_b)
            self.levels = int(float(pe_embed_l))
            self.embed_length = 2 * self.levels
            self.pe_embed = True

    def __repr__(self):
        return f"Positional Encoder: pos_b={self.lbase}, pos_l={self.levels}, embed_length={self.embed_length}, to_embed={self.pe_embed}"

    def forward(self, pos):
        if not self.pe_embed:
            return pos[:, None]
        else:
            pe_list = []
            #pos = pos.permute(1, 0, 2)
            for i in range(self.levels):
                temp_value = pos * (self.lbase ** i) * math.pi
                pe_list += [torch.sin(temp_value), torch.cos(temp_value)]
            return torch.cat(pe_list, 1)
     
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
        else:
            self.skip = None

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        return self.body(x) + identity
        

class JointContextTransfer(nn.Module):
    def __init__(self, channels):
        super(JointContextTransfer, self).__init__()
        self.rb1 = ResidualBlock(channels, channels)
        self.rb2 = ResidualBlock(channels, channels)
        self.attn = EfficientAttention(key_in_channels=channels, query_in_channels=channels, key_channels=channels//6, 
            head_count=2, value_channels=channels//6)

        self.refine = nn.Sequential(
            ResidualBlock(channels*2, channels),
            ResidualBlock(channels, channels))

    def forward(self, x_left, x_right):
        B, C, H, W = x_left.size()
        identity_left, identity_right = x_left, x_right
        x_left, x_right = self.rb2(self.rb1(x_left)), self.rb2(self.rb1(x_right))
        
        #print(x_left.shape, x_right.shape)
        A_right_to_left, A_left_to_right = self.attn.parallel_forward(x_left, x_right), self.attn.parallel_forward(x_right, x_left)
        compact_left = identity_left + self.refine(torch.cat((A_right_to_left, x_left), dim=1))
        compact_right = identity_right + self.refine(torch.cat((A_left_to_right, x_right), dim=1))
        return compact_left, compact_right#, A_right_to_left, A_left_to_right


class EfficientAttention(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, key_channels=32, head_count=8, value_channels=64):
        super().__init__()
        self.in_channels = query_in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(key_in_channels, key_channels, 1)
        self.queries = nn.Conv2d(query_in_channels, key_channels, 1)
        self.values = nn.Conv2d(key_in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, query_in_channels, 1)

    def forward(self, target, input):
        n, _, h, w = input.size()
        keys = self.keys(input).reshape((n, self.key_channels, h * w))
        queries = self.queries(target).reshape(n, self.key_channels, h * w)
        values = self.values(input).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels,:], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels,:], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value #+ input_

        return attention

    def parallel_forward(self, target, input):
        n, _, h, w = input.size()
        keys = self.keys(input).reshape((n, self.key_channels, h * w))
        queries = self.queries(target).reshape(n, self.key_channels, h * w)
        values = self.values(input).reshape((n, self.value_channels, h * w))
        
        keys = keys.permute(0, 2, 1).reshape(n, h*w, self.head_count, -1).permute(0, 2, 3, 1).reshape(n*self.head_count, -1, h*w)
        queries = queries.permute(0, 2, 1).reshape(n, h*w, self.head_count, -1).permute(0, 2, 3, 1).reshape(n*self.head_count, -1, h*w)
        values = values.permute(0, 2, 1).reshape(n, h*w, self.head_count, -1).permute(0, 2, 3, 1).reshape(n*self.head_count, -1, h*w)

        keys = F.softmax(keys, dim=2)
        queries = F.softmax(queries, dim=1)
        context = keys @ values.transpose(1, 2)
        aggregated_values = (context.transpose(1, 2) @ queries).reshape(n, -1, h, w)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value #+ target
        return attention       
            
class StereoBaseline(nn.Module):
    """
    Baseline model for stereo image compression. Encoder/decoder with hyperprior entropy model. 
    Left and right image are compressed separately. Is used as base class for ECSIC.
    """

    def __init__(self, in_channels=3, N=192, M=12):
        super().__init__()
        
        self.E = StereoSequential(
            Stereo(nn.Conv2d, in_channels, N, 3, 2, 1),					                # 1
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.Conv2d, N, N, 3, 2, 1),							                # 2
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.Conv2d, N, N, 3, 2, 1),							                # 4
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.Conv2d, N, M, 3, 1, 1)							                # 8
        )

        self.D = StereoSequential(
            Stereo(nn.ConvTranspose2d, M, N, 3, 2, 1, 1),				                # 8
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.ConvTranspose2d, N, N, 3, 2, 1, 1),				                # 4
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.ConvTranspose2d, N, N, 3, 2, 1, 1),				                # 2
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.ConvTranspose2d, N, in_channels, 3, 1, 1, 0)                      # 1
        )
    
        self.HE = StereoSequential(
            Stereo(nn.Conv2d, M, N, 3, 2, 1),							                # 8
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.Conv2d, N, N, 3, 2, 1),							                # 16
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.Conv2d, N, M, 3, 1, 1)							                # 32
        )

        self.HD = StereoSequential(
            Stereo(nn.ConvTranspose2d, M, N, 3, 2, 1, 1),				                # 32
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.ConvTranspose2d, N, N, 3, 2, 1, 1),				                # 16
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.Conv2d, N, 2*M, 3, 1, 1),				                            # 8
        )
        
        self.pe = PositionalEncoding(pe_embed_b=1.25, pe_embed_l=8)

        '''self.zl_loc = nn.Parameter(torch.empty((1, M, 1, 1)))
        self.zl_loc.data.normal_(0.0, 1.0)

        self.zl_scale = nn.Parameter(torch.empty((1, M, 1, 1)))
        self.zl_scale.data.uniform_(1.0, 1.5)

        self.zr_loc = nn.Parameter(torch.empty((1, M, 1, 1)))
        self.zr_loc.data.normal_(0.0, 1.0)

        self.zr_scale = nn.Parameter(torch.empty((1, M, 1, 1)))
        self.zr_scale.data.uniform_(1.0, 1.5)'''
        
        self.zl_fc = nn.Sequential(
                nn.Linear(3*2*8, N),
                nn.LeakyReLU(),
                nn.Linear(N, 2*M)
            )
            
        self.zr_fc = nn.Sequential(
                nn.Linear(3*2*8, N),
                nn.LeakyReLU(),
                nn.Linear(N, 2*M)
            )

    def entropy(self, yl, yr, pl, pr, pos=None):
        zl, zr, ahe_rtl, ahe_ltr = self.HE(yl, yr, pos=pos, return_attn=True)     

        # Quantise z
        zl_loc, zl_scale = self.zl_fc(pl).chunk(2, 1)
        zl_loc = zl_loc.unsqueeze(-1).unsqueeze(-1)
        zl_scale = zl_scale.unsqueeze(-1).unsqueeze(-1)
        zr_loc, zr_scale = self.zr_fc(pr).chunk(2, 1)
        zr_loc = zr_loc.unsqueeze(-1).unsqueeze(-1)
        zr_scale = zr_scale.unsqueeze(-1).unsqueeze(-1)
        zl_hat_ent, zl_hat_dec = quantise(zl, zl_loc, training=self.training)
        zr_hat_ent, zr_hat_dec = quantise(zr, zr_loc, training=self.training)

        # Compute probability parameters for y
        yl_entropy, yr_entropy, ahd_rtl, ahd_ltr = self.HD(zl_hat_dec, zr_hat_dec, pos=pos, return_attn=True)
        yl_loc, yl_scale = torch.chunk(yl_entropy, chunks=2, dim=1)
        yr_loc, yr_scale = torch.chunk(yr_entropy, chunks=2, dim=1)

        # Quantise y
        yl_hat_ent, yl_hat_dec = quantise(yl, yl_loc, training=self.training)
        yr_hat_ent, yr_hat_dec = quantise(yr, yr_loc, training=self.training)

        latents = Dict(
            left = Dict(
                y_hat_ent=yl_hat_ent,
                y_hat_dec=yl_hat_dec,
                y_loc=yl_loc,
                y_scale=yl_scale,

                z_hat_ent=zl_hat_ent,
                z_hat_dec=zl_hat_dec,
                z_loc=zl_loc,
                z_scale=zl_scale
            ),
            right = Dict(
                y_hat_ent=yr_hat_ent,
                y_hat_dec=yr_hat_dec,
                y_loc=yr_loc,
                y_scale=yr_scale,

                z_hat_ent=zr_hat_ent,
                z_hat_dec=zr_hat_dec,
                z_loc=zr_loc,
                z_scale=zr_scale
            )
        )

        return latents

    def forward(self, xl, xr, pl, pr, pos=None):
        # forward pass through model
        pl_pe = self.pe(pl.squeeze(1))
        pr_pe = self.pe(pr.squeeze(1))
        yl, yr, ae_rtl, ae_ltr = self.E(xl, xr, pos=pos, return_attn=True)
        latents = self.entropy(yl, yr, pl_pe, pr_pe, pos)
        xl_hat, xr_hat, ad_rtl, ad_ltr = self.D(latents.left.y_hat_dec, latents.right.y_hat_dec, pos=pos, return_attn=True)

        # Calculate rates for z and y
        bpp_zl = calc_rate(latents.left.z_hat_ent, latents.left.z_loc, latents.left.z_scale)
        bpp_zr = calc_rate(latents.right.z_hat_ent, latents.right.z_loc, latents.right.z_scale)
        bpp_yl = calc_rate(latents.left.y_hat_ent, latents.left.y_loc, latents.left.y_scale)
        bpp_yr = calc_rate(latents.right.y_hat_ent, latents.right.y_loc, latents.right.y_scale)

        return Dict(
            latents=latents,
            rate=Dict(
                left = Dict(
                    y=bpp_yl,
                    z=bpp_zl
                ),
                right = Dict(
                    y=bpp_yr,
                    z=bpp_zr
                )
            ),
            pred=Dict(
                left=xl_hat,
                right=xr_hat
            )
        )


class StereoAttentionModelPlus(StereoBaseline):

    def __init__(self, in_channels=3, N=192, M=12, z_context=True, y_context=True, attn_mask=False, pos_encoding=False, 
                 ln=True, shared=True, ff=True, valid_mask=None, rel_pos_enc=False, embed=None, heads=4, only_D=False):
        super().__init__(in_channels=in_channels, N=N, M=M)

        self.z_context = z_context
        self.y_context = y_context
        args = {
            'attn_mask': attn_mask,
            'pos_encoding': pos_encoding,
            'ln': ln,
            'ff': ff,
            'valid_mask': valid_mask,
            'rel_pos_enc': rel_pos_enc
        }
        embed = 2*N if embed is None else embed
        heads = heads

 
        self.E = StereoSequential(
            Stereo(nn.Conv2d, in_channels, N, 3, 2, 1, shared=shared),					# 1
            Stereo(nn.PReLU, N, init=0.2, shared=shared),
            Stereo(nn.Conv2d, N, N, 3, 2, 1, shared=shared),							# 2
            Stereo(nn.PReLU, N, init=0.2, shared=shared),
            Stereo(nn.Conv2d, N, N, 3, 2, 1, shared=shared),							# 4
            Stereo(nn.PReLU, N, init=0.2, shared=shared),
            JointContextTransfer(N),	                        # 8
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.Conv2d, N, M, 3, 1, 1)							                # 8
        )

        self.D = StereoSequential(
            Stereo(nn.ConvTranspose2d, M, N, 3, 2, 1, 1),				                # 8
            Stereo(nn.PReLU, N, init=0.2),
            JointContextTransfer(N),	                        # 8
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.ConvTranspose2d, N, N, 3, 2, 1, 1),				                # 4
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.ConvTranspose2d, N, N, 3, 2, 1, 1),				                # 2
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.ConvTranspose2d, N, in_channels, 3, 1, 1, 0)                      # 1
        )
    
        self.HE = StereoSequential(
            Stereo(nn.Conv2d, M, N, 3, 2, 1),							                # 8
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.Conv2d, N, N, 3, 2, 1),							                # 16
            Stereo(nn.PReLU, N, init=0.2),
            JointContextTransfer(N),		                    # 16
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.Conv2d, N, M, 3, 1, 1)							                # 32
        )

        self.HD = StereoSequential(
            Stereo(nn.ConvTranspose2d, M, N, 3, 2, 1, 1),				                # 32
            Stereo(nn.PReLU, N, init=0.2),
            JointContextTransfer(N),		                    # 16
            Stereo(nn.PReLU, N, init=0.2),
            Stereo(nn.ConvTranspose2d, N, N, 3, 2, 1, 1),				                # 16
            Stereo(nn.PReLU, N, init=0.2),
        )
        self.hd_out_left = nn.Conv2d(N, 2*M, 3, 1, 1)

        if self.z_context:
            self.zC = nn.Sequential(
                nn.Conv2d(3*M, 4*M, 3, 1, 1),
                nn.PReLU(4*M, init=0.2),
                nn.Conv2d(4*M, 4*M, 3, 1, 1),
                nn.PReLU(4*M, init=0.2),
                nn.Conv2d(4*M, 2*M, 3, 1, 1)
            )

        if self.y_context:
            self.hd_out_right = nn.Conv2d(N, N, 3, 1, 1)

            self.yle_conv = nn.Conv2d(M, N, 3, 1, 1)
            self.yle_prelu = nn.PReLU(N, init=0.2)
            self.yre_conv = nn.Conv2d(N, N, 3, 1, 1)
            self.yre_prelu = nn.PReLU(N, init=0.2)
            self.ye_sa = JointContextTransfer(N)
            self.ye_prelu = nn.PReLU(2*N, init=0.2)
            self.ye_conv = nn.Conv2d(2*N, 2*M, 3, 1, 1)
        else:
            self.hd_out_right = nn.Conv2d(N, 2*M, 3, 1, 1)

    def yr_entropy(self, yl_hat, yr_entropy, pos):
        yle = self.yle_prelu(self.yle_conv(yl_hat))
        yre = self.yre_prelu(self.yre_conv(yr_entropy))
        yle, yre = self.ye_sa(yle, yre)
        ye = self.ye_prelu(torch.cat([yle, yre], dim=1))
        ye = self.ye_conv(ye)

        yr_loc, yr_scale = ye.chunk(2, 1)

        return yr_loc, yr_scale, 0, 0
    
    def zr_entropy(self, zl_hat, pr):
        s = zl_hat.shape
        
        zr_loc, zr_scale = self.zr_fc(pr).chunk(2, 1)
        zr_loc = zr_loc.unsqueeze(-1).unsqueeze(-1)
        zr_scale = zr_scale.unsqueeze(-1).unsqueeze(-1)
        zC_in = torch.cat([zr_loc.expand(s), zr_scale.expand(s), zl_hat], dim=1)
        zC_out = self.zC(zC_in)
        zr_loc, zr_scale = zC_out.chunk(2, 1)
        zr_scale = F.relu(zr_scale)

        return zr_loc, zr_scale
        
    '''def zr_entropy(self, zl_hat):
        s = zl_hat.shape

        zC_in = torch.cat([self.zr_loc.expand(s), self.zr_scale.expand(s), zl_hat], dim=1)
        zC_out = self.zC(zC_in)
        zr_loc, zr_scale = zC_out.chunk(2, 1)
        zr_scale = F.relu(zr_scale)

        return zr_loc, zr_scale'''

    def entropy(self, yl, yr, pl, pr, pos=None):
        zl, zr, ahe_rtl, ahe_ltr = self.HE(yl, yr, pos=pos, return_attn=True)     

        # Quantise z
        zl_loc, zl_scale = self.zl_fc(pl).chunk(2, 1)
        zl_loc = zl_loc.unsqueeze(-1).unsqueeze(-1)
        zl_scale = zl_scale.unsqueeze(-1).unsqueeze(-1)
        zl_hat_ent, zl_hat_dec = quantise(zl, zl_loc, training=self.training)

        #if self.z_context:
        zr_loc, zr_scale = self.zr_entropy(zl_hat_dec, pr)
        #else:
        #    zr_loc, zr_scale = self.zr_loc, self.zr_scale
        zr_hat_ent, zr_hat_dec = quantise(zr, zr_loc, training=self.training)

        # Compute probability parameters for y
        yl_entropy, yr_entropy, ahd_rtl, ahd_ltr = self.HD(zl_hat_dec, zr_hat_dec, pos=pos, return_attn=True)
        yl_entropy = self.hd_out_left(yl_entropy)
        yr_entropy = self.hd_out_right(yr_entropy)
        yl_loc, yl_scale = torch.chunk(yl_entropy, chunks=2, dim=1)

        # Quantise y left
        yl_hat_ent, yl_hat_dec = quantise(yl, yl_loc, training=self.training)

        # Compute probability parameters for y right
        if self.y_context:
            yr_loc, yr_scale, ay_rtl, ay_ltr = self.yr_entropy(yl_hat_dec, yr_entropy, pos=pos)
        else:
            yr_loc, yr_scale = torch.chunk(yr_entropy, chunks=2, dim=1)
            ay_rtl, ay_ltr = [], []
        yr_hat_ent, yr_hat_dec = quantise(yr, yr_loc, training=self.training)

        latents = Dict(
            left = Dict(
                y_hat_ent=yl_hat_ent,
                y_hat_dec=yl_hat_dec,
                y_loc=yl_loc,
                y_scale=yl_scale,

                z_hat_ent=zl_hat_ent,
                z_hat_dec=zl_hat_dec,
                z_loc=zl_loc,
                z_scale=zl_scale
            ),
            right = Dict(
                y_hat_ent=yr_hat_ent,
                y_hat_dec=yr_hat_dec,
                y_loc=yr_loc,
                y_scale=yr_scale,

                z_hat_ent=zr_hat_ent,
                z_hat_dec=zr_hat_dec,
                z_loc=zr_loc,
                z_scale=zr_scale
            )
        )

        return latents


class DebugModel(StereoBaseline):

    def __init__(self, in_channels=3, M=6):
        super().__init__(in_channels=in_channels, M=M)
        s = 'DEBUG MODEL IS USED!'
        print('#'*(len(s) + 4) + f'\n# {s} #\n' + '#'*(len(s) + 4))

        self.E = Stereo(nn.Conv2d, in_channels, M, 3, 1, 1)
        self.D = Stereo(nn.Conv2d, M, in_channels, 3, 1, 1)
        self.HE = Stereo(nn.Conv2d, M, M, 3, 1, 1)
        self.HD = Stereo(nn.Conv2d, M, 2*M, 3, 1, 1)