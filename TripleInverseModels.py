#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exp35 three-parameter inverse models.

All models expose the same interface:
    set_input_normalization(mean, scale, global_rms=None)
    forward(g) -> (a1, m, log10(gamma))

Models
------
1. TripleInverseMLP
2. TripleInverseTCN
3. TripleInverseParameterSpecific

The parameter-specific model encodes the physical lesson from Exp33/34:
- gamma and m rely mainly on local/global SHAPE information from a TCN encoder;
- a1 additionally receives an amplitude-preserving raw-g branch.

No analytic parameter solve is used.  All three parameters are direct neural
network outputs.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels, preferred=8):
    for g in range(min(int(preferred),int(channels)),0,-1):
        if channels % g == 0:
            return g
    return 1


class DilatedResidualBlock1D(nn.Module):
    def __init__(self, channels, dilation, kernel_size=3, dropout=0.05):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        pad = int(dilation)*(int(kernel_size)-1)//2
        self.conv1=nn.Conv1d(channels,channels,kernel_size,padding=pad,dilation=dilation)
        self.norm1=nn.GroupNorm(_groups(channels),channels)
        self.conv2=nn.Conv1d(channels,channels,kernel_size,padding=pad,dilation=dilation)
        self.norm2=nn.GroupNorm(_groups(channels),channels)
        self.act=nn.GELU(); self.drop=nn.Dropout(float(dropout))
    def forward(self,x):
        r=x
        x=self.act(self.norm1(self.conv1(x))); x=self.drop(x)
        x=self.norm2(self.conv2(x)); x=self.drop(x)
        return self.act(x+r)


class _BoundsMixin:
    def _init_bounds(self,input_points,a1_min,a1_max,m_min,m_max,gamma_min,gamma_max):
        self.input_points=int(input_points)
        if not (a1_min<a1_max and m_min<m_max and 0<gamma_min<gamma_max):
            raise ValueError("invalid parameter bounds")
        self.register_buffer("input_mean",torch.zeros(self.input_points))
        self.register_buffer("input_scale",torch.ones(self.input_points))
        self.register_buffer("global_input_scale",torch.tensor(1.0))
        self.register_buffer("_a1_min",torch.tensor(float(a1_min)))
        self.register_buffer("_a1_max",torch.tensor(float(a1_max)))
        self.register_buffer("_m_min",torch.tensor(float(m_min)))
        self.register_buffer("_m_max",torch.tensor(float(m_max)))
        self.register_buffer("_lg_min",torch.tensor(math.log10(float(gamma_min))))
        self.register_buffer("_lg_max",torch.tensor(math.log10(float(gamma_max))))
    @torch.no_grad()
    def set_input_normalization(self,mean,scale,global_rms=None):
        mean=torch.as_tensor(mean,dtype=self.input_mean.dtype).reshape(-1)
        scale=torch.as_tensor(scale,dtype=self.input_scale.dtype).reshape(-1)
        if mean.numel()!=self.input_points or scale.numel()!=self.input_points:
            raise ValueError("mean/scale length mismatch")
        if torch.any(scale<=0) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
            raise ValueError("invalid normalization")
        self.input_mean.copy_(mean.to(self.input_mean.device))
        self.input_scale.copy_(scale.to(self.input_scale.device))
        if global_rms is not None:
            gr=torch.as_tensor(global_rms,dtype=self.global_input_scale.dtype).reshape(())
            if not torch.isfinite(gr) or float(gr)<=0:
                raise ValueError("invalid global_rms")
            self.global_input_scale.copy_(gr.to(self.global_input_scale.device))
    def _coerce(self,gy):
        if gy.ndim==3:
            if gy.shape[1]!=1: raise ValueError("expected [B,1,N]")
            gy=gy[:,0,:]
        if gy.ndim!=2 or gy.shape[1]!=self.input_points:
            raise ValueError("expected [B,%d]"%self.input_points)
        return gy
    def _standardized(self,gy):
        gy=self._coerce(gy)
        mean=self.input_mean.to(gy); scale=self.input_scale.to(gy)
        return (gy-mean)/scale
    def _decode(self,logits):
        a1u=torch.sigmoid(logits[:,0]); mu=torch.sigmoid(logits[:,1]); gu=torch.sigmoid(logits[:,2])
        a1=self._a1_min.to(a1u)+a1u*(self._a1_max.to(a1u)-self._a1_min.to(a1u))
        m=self._m_min.to(mu)+mu*(self._m_max.to(mu)-self._m_min.to(mu))
        lg=self._lg_min.to(gu)+gu*(self._lg_max.to(gu)-self._lg_min.to(gu))
        return a1,m,lg


class TripleInverseMLP(nn.Module,_BoundsMixin):
    def __init__(self,*,input_points=1000,hidden_sizes=(256,256,128),dropout=.05,
                 a1_min,a1_max,m_min,m_max,gamma_min,gamma_max):
        super().__init__(); self._init_bounds(input_points,a1_min,a1_max,m_min,m_max,gamma_min,gamma_max)
        layers=[]; d=self.input_points
        for h in hidden_sizes:
            layers += [nn.Linear(d,int(h)),nn.GELU(),nn.Dropout(float(dropout))]
            d=int(h)
        self.backbone=nn.Sequential(*layers); self.head=nn.Linear(d,3)
    def forward(self,gy):
        return self._decode(self.head(self.backbone(self._standardized(gy))))


class _TCNEncoder(nn.Module):
    def __init__(self,input_points,channels,dilations,kernel_size,dropout,use_coordinate_channel):
        super().__init__()
        self.input_points=int(input_points); self.use_coordinate_channel=bool(use_coordinate_channel)
        coord=torch.linspace(-1,1,self.input_points).view(1,1,-1)
        self.register_buffer("coordinate",coord)
        inc=2 if self.use_coordinate_channel else 1
        self.stem=nn.Sequential(
            nn.Conv1d(inc,int(channels),5,padding=2),
            nn.GroupNorm(_groups(int(channels)),int(channels)),nn.GELU()
        )
        self.blocks=nn.Sequential(*[
            DilatedResidualBlock1D(int(channels),int(d),kernel_size=int(kernel_size),dropout=float(dropout))
            for d in dilations
        ])
        self.channels=int(channels)
    def forward(self,x_standardized):
        x=x_standardized.unsqueeze(1)
        if self.use_coordinate_channel:
            c=self.coordinate.to(x).expand(x.shape[0],-1,-1)
            x=torch.cat([x,c],dim=1)
        x=self.blocks(self.stem(x))
        return torch.cat([torch.mean(x,dim=2),torch.amax(x,dim=2)],dim=1)


class TripleInverseTCN(nn.Module,_BoundsMixin):
    def __init__(self,*,input_points=1000,channels=64,dilations=(1,2,4,8,16),
                 kernel_size=3,dropout=.05,head_hidden=128,use_coordinate_channel=True,
                 a1_min,a1_max,m_min,m_max,gamma_min,gamma_max):
        super().__init__(); self._init_bounds(input_points,a1_min,a1_max,m_min,m_max,gamma_min,gamma_max)
        self.encoder=_TCNEncoder(input_points,channels,dilations,kernel_size,dropout,use_coordinate_channel)
        self.head=nn.Sequential(
            nn.Linear(2*int(channels),int(head_hidden)),nn.GELU(),nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden),int(head_hidden)),nn.GELU(),nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden),3)
        )
    def forward(self,gy):
        z=self.encoder(self._standardized(gy))
        return self._decode(self.head(z))


class TripleInverseParameterSpecific(nn.Module,_BoundsMixin):
    """Shared TCN shape encoder + amplitude-aware a1 branch + separate heads."""
    def __init__(self,*,input_points=1000,channels=64,dilations=(1,2,4,8,16),
                 kernel_size=3,dropout=.05,head_hidden=128,use_coordinate_channel=True,
                 amplitude_bins=16,amplitude_hidden=96,
                 a1_min,a1_max,m_min,m_max,gamma_min,gamma_max):
        super().__init__(); self._init_bounds(input_points,a1_min,a1_max,m_min,m_max,gamma_min,gamma_max)
        self.encoder=_TCNEncoder(input_points,channels,dilations,kernel_size,dropout,use_coordinate_channel)
        self.amplitude_bins=int(amplitude_bins)
        shape_dim=2*int(channels)
        # avg+max pooled globally-scaled raw signal + 8 scalar amplitude stats
        amp_in=2*self.amplitude_bins+8
        self.amp_encoder=nn.Sequential(
            nn.Linear(amp_in,int(amplitude_hidden)),nn.GELU(),nn.Dropout(float(dropout)),
            nn.Linear(int(amplitude_hidden),int(amplitude_hidden)),nn.GELU()
        )
        self.a1_head=nn.Sequential(
            nn.Linear(shape_dim+int(amplitude_hidden),int(head_hidden)),nn.GELU(),
            nn.Dropout(float(dropout)),nn.Linear(int(head_hidden),1)
        )
        self.m_head=nn.Sequential(
            nn.Linear(shape_dim,int(head_hidden)),nn.GELU(),nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden),1)
        )
        self.gamma_head=nn.Sequential(
            nn.Linear(shape_dim,int(head_hidden)),nn.GELU(),nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden),1)
        )
    def _amp_features(self,gy):
        gy=self._coerce(gy)
        mean=self.input_mean.to(gy)
        gs=self.global_input_scale.to(gy).clamp_min(torch.finfo(gy.dtype).eps)
        x=(gy-mean)/gs
        x1=x.unsqueeze(1)
        av=F.adaptive_avg_pool1d(x1,self.amplitude_bins).squeeze(1)
        mx=F.adaptive_max_pool1d(x1,self.amplitude_bins).squeeze(1)
        meanx=x.mean(1); std=x.std(1,unbiased=False)
        rms=torch.sqrt(torch.mean(x*x,1).clamp_min(1e-12))
        ma=torch.mean(torch.abs(x),1)
        xmax=torch.amax(x,1); xmin=torch.amin(x,1); ptp=xmax-xmin
        rawrms=torch.sqrt(torch.mean(gy*gy,1).clamp_min(1e-12))
        logr=torch.log(rawrms/gs+1e-12)
        stats=torch.stack([meanx,std,rms,ma,xmax,xmin,ptp,logr],1)
        return torch.cat([av,mx,stats],1)
    def forward(self,gy):
        shape=self.encoder(self._standardized(gy))
        amp=self.amp_encoder(self._amp_features(gy))
        a1logit=self.a1_head(torch.cat([shape,amp],1)).squeeze(1)
        mlogit=self.m_head(shape).squeeze(1)
        glogit=self.gamma_head(shape).squeeze(1)
        logits=torch.stack([a1logit,mlogit,glogit],1)
        return self._decode(logits)
