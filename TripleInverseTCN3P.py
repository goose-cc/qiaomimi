#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Self-contained TCN for Exp36 3P inverse regression.

Input : gy [B, N]
Output: a1, m, log10(gamma)
"""
from __future__ import annotations
import math
from typing import Sequence
import torch
import torch.nn as nn


def _groups(c:int, preferred:int=8)->int:
    for g in range(min(preferred,c),0,-1):
        if c%g==0: return g
    return 1


class ResidualBlock(nn.Module):
    def __init__(self,c:int,dilation:int,kernel_size:int=3,dropout:float=0.05):
        super().__init__()
        if kernel_size%2!=1: raise ValueError("kernel_size must be odd")
        pad=dilation*(kernel_size-1)//2
        g=_groups(c)
        self.conv1=nn.Conv1d(c,c,kernel_size,padding=pad,dilation=dilation)
        self.norm1=nn.GroupNorm(g,c)
        self.conv2=nn.Conv1d(c,c,kernel_size,padding=pad,dilation=dilation)
        self.norm2=nn.GroupNorm(g,c)
        self.act=nn.GELU()
        self.drop=nn.Dropout(dropout)
    def forward(self,x):
        r=x
        x=self.drop(self.act(self.norm1(self.conv1(x))))
        x=self.drop(self.norm2(self.conv2(x)))
        return self.act(x+r)


class TripleInverseTCN3P(nn.Module):
    def __init__(
        self,input_points:int=100,channels:int=64,dilations:Sequence[int]=(1,2,4,8,16),
        kernel_size:int=3,head_hidden:int=160,dropout:float=0.05,
        a1_range=(0.05,0.20),m_range=(0.40,1.20),gamma_range=(0.01,1.0),
        coordinate_channel:bool=True,
    ):
        super().__init__()
        self.input_points=int(input_points)
        self.coordinate_channel=bool(coordinate_channel)
        self.register_buffer("input_mean",torch.zeros(self.input_points))
        self.register_buffer("input_scale",torch.ones(self.input_points))
        self.register_buffer("coord",torch.linspace(-1,1,self.input_points).view(1,1,-1))
        self.register_buffer("a1_lo",torch.tensor(float(a1_range[0])))
        self.register_buffer("a1_hi",torch.tensor(float(a1_range[1])))
        self.register_buffer("m_lo",torch.tensor(float(m_range[0])))
        self.register_buffer("m_hi",torch.tensor(float(m_range[1])))
        self.register_buffer("lg_lo",torch.tensor(math.log10(float(gamma_range[0]))))
        self.register_buffer("lg_hi",torch.tensor(math.log10(float(gamma_range[1]))))
        inc=2 if self.coordinate_channel else 1
        self.stem=nn.Sequential(
            nn.Conv1d(inc,channels,5,padding=2),
            nn.GroupNorm(_groups(channels),channels),nn.GELU()
        )
        self.blocks=nn.Sequential(*[ResidualBlock(channels,int(d),kernel_size,dropout) for d in dilations])
        self.head=nn.Sequential(
            nn.Linear(2*channels,head_hidden),nn.GELU(),nn.Dropout(dropout),
            nn.Linear(head_hidden,head_hidden),nn.GELU(),nn.Dropout(dropout),
            nn.Linear(head_hidden,3),
        )
    @torch.no_grad()
    def set_input_normalization(self,mean,scale):
        mean=torch.as_tensor(mean,dtype=self.input_mean.dtype).reshape(-1)
        scale=torch.as_tensor(scale,dtype=self.input_scale.dtype).reshape(-1)
        if len(mean)!=self.input_points or len(scale)!=self.input_points: raise ValueError("normalization length mismatch")
        if torch.any(scale<=0) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all(): raise ValueError("invalid normalization")
        self.input_mean.copy_(mean); self.input_scale.copy_(scale)
    def _input(self,gy):
        if gy.ndim==3: gy=gy[:,0,:]
        if gy.ndim!=2 or gy.shape[1]!=self.input_points: raise ValueError(f"expected [B,{self.input_points}]")
        mean=self.input_mean.to(gy); scale=self.input_scale.to(gy)
        x=((gy-mean)/scale).unsqueeze(1)
        if self.coordinate_channel:
            x=torch.cat([x,self.coord.to(x).expand(x.shape[0],-1,-1)],dim=1)
        return x
    def forward(self,gy):
        x=self.blocks(self.stem(self._input(gy)))
        z=torch.cat([x.mean(dim=2),x.amax(dim=2)],dim=1)
        logits=self.head(z)
        u=torch.sigmoid(logits)
        a1=self.a1_lo.to(u)+u[:,0]*(self.a1_hi-self.a1_lo).to(u)
        m=self.m_lo.to(u)+u[:,1]*(self.m_hi-self.m_lo).to(u)
        lg=self.lg_lo.to(u)+u[:,2]*(self.lg_hi-self.lg_lo).to(u)
        return a1,m,lg
    def predict_physical(self,gy):
        a1,m,lg=self.forward(gy)
        return a1,m,torch.pow(10.0,lg)
