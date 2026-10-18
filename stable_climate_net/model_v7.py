"""V3rc2: stronger permutation-equivariant set-temporal backbone plus symmetric spatial residual.

The temporal backbone uses one shared causal TCN for every station and permutation-
invariant pooling across stations. The target-station embedding is gathered after the
shared encoder. Spatial add-ons are trained on top of a frozen set-temporal backbone.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_v6 import pairwise_haversine_km, operator_errors

class SharedCausalStationEncoder(nn.Module):
    def __init__(self, n_features:int, hidden:int=48, dropout:float=0.05):
        super().__init__()
        self.inp=nn.Conv1d(n_features,hidden,1)
        self.blocks=nn.ModuleList([nn.Conv1d(hidden,hidden,3,dilation=d) for d in (1,2,4)])
        self.norms=nn.ModuleList([nn.GroupNorm(1,hidden) for _ in range(3)])
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        B,T,S,Ftr=x.shape
        y=torch.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0).permute(0,2,3,1).reshape(B*S,Ftr,T)
        y=self.inp(y)
        for conv,norm,d in zip(self.blocks,self.norms,(1,2,4)):
            r=conv(F.pad(y,(2*d,0)))
            y=y+self.dropout(F.gelu(norm(r)))
        return y[:,:,-1].reshape(B,S,-1)

class SetTemporalSymmetricSpatialNet(nn.Module):
    MODES=(
        'set_temporal_raw_control',
        'set_temporal_plus_symmetric_spatial',
        'set_temporal_plus_sym_asym_spatial',
        'set_temporal_plus_unconstrained_spatial',
    )
    def __init__(self,n_features:int,station_coordinates,history:int=30,hidden:int=48,head_hidden:int=64,
                 message_dim:int=12,n_heads:int=4,aux_hidden:int=32,quantiles=(0.05,0.50,0.95),
                 target_feature_index:int=0,mode:str='set_temporal_plus_symmetric_spatial'):
        super().__init__()
        if mode not in self.MODES: raise ValueError(f'unknown mode {mode}')
        coords=torch.as_tensor(station_coordinates,dtype=torch.float32)
        if coords.ndim!=2 or coords.shape[1]!=2 or not torch.isfinite(coords).all():
            raise ValueError('finite station_coordinates [S,2] required')
        self.register_buffer('station_coordinates',coords.clone())
        self.n_stations=int(coords.shape[0]); self.n_features=int(n_features); self.history=int(history)
        self.hidden=int(hidden); self.message_dim=int(message_dim); self.n_heads=int(n_heads)
        self.quantiles=tuple(quantiles); self.target_feature_index=int(target_feature_index); self.mode=str(mode)
        self.station_temporal=SharedCausalStationEncoder(n_features,hidden)
        self.raw_head=nn.Sequential(nn.Linear(3*hidden,head_hidden),nn.GELU(),nn.Linear(head_hidden,1))
        self.width_head=nn.Sequential(nn.Linear(3*hidden,head_hidden),nn.GELU(),nn.Linear(head_hidden,2))
        if mode!='set_temporal_raw_control':
            init_scales=torch.linspace(120.0,650.0,n_heads)
            self.log_length_km=nn.Parameter(torch.log(torch.expm1(init_scales.clamp_min(1.0))))
            if mode in ('set_temporal_plus_symmetric_spatial','set_temporal_plus_sym_asym_spatial'):
                self.sym_affinity=nn.Linear(hidden,n_heads,bias=False)
                self.sym_value=nn.Linear(hidden,n_heads*message_dim,bias=False)
                self.sym_head=nn.Sequential(nn.Linear(hidden+n_heads*message_dim,aux_hidden),nn.GELU(),nn.Linear(aux_hidden,1))
            if mode=='set_temporal_plus_sym_asym_spatial':
                self.direction_potential=nn.Linear(hidden,n_heads,bias=False)
                self.asym_value=nn.Linear(hidden,n_heads*message_dim,bias=False)
                self.asym_head=nn.Sequential(nn.Linear(hidden+n_heads*message_dim,aux_hidden),nn.GELU(),nn.Linear(aux_hidden,1))
            if mode=='set_temporal_plus_unconstrained_spatial':
                self.unconstrained_edge=nn.Sequential(nn.Linear(2*hidden+1,aux_hidden),nn.GELU(),nn.Linear(aux_hidden,n_heads))
                self.unconstrained_value=nn.Linear(hidden,n_heads*message_dim,bias=False)
                self.unconstrained_head=nn.Sequential(nn.Linear(hidden+n_heads*message_dim,aux_hidden),nn.GELU(),nn.Linear(aux_hidden,1))
        nn.init.zeros_(self.raw_head[-1].weight); nn.init.zeros_(self.raw_head[-1].bias)
        nn.init.zeros_(self.width_head[-1].weight); nn.init.zeros_(self.width_head[-1].bias)
        for name in ('sym_head','asym_head','unconstrained_head'):
            m=getattr(self,name,None)
            if m is not None:
                nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)

    @staticmethod
    def _indices(B,target_station_index,device):
        if target_station_index is None: raise ValueError('target_station_index required')
        if isinstance(target_station_index,int): return torch.full((B,),target_station_index,device=device,dtype=torch.long)
        return target_station_index.to(device=device,dtype=torch.long).reshape(B)
    @staticmethod
    def _gather(z,idx):
        B,S,D=z.shape
        return z.gather(1,idx[:,None,None].expand(B,1,D)).squeeze(1)
    def _support(self,coords,B):
        dist=pairwise_haversine_km(coords)
        scales=F.softplus(self.log_length_km).clamp_min(1.0)
        K=torch.exp(-dist[None]/scales[:,None,None])
        eye=torch.eye(dist.shape[0],device=dist.device,dtype=dist.dtype)[None]
        K=K*(1-eye)
        return K[None].expand(B,-1,-1,-1),dist
    def _spatial(self,z,idx,coords):
        B,S,D=z.shape; zt=self._gather(z,idx); norm=float(max(S-1,1))
        zero=z.new_zeros(B); sym=zero; asym=zero; unc=zero
        Sop=z.new_zeros(B,self.n_heads,S,S); Aop=torch.zeros_like(Sop)
        dist=pairwise_haversine_km(coords)
        if self.mode in ('set_temporal_plus_symmetric_spatial','set_temporal_plus_sym_asym_spatial'):
            K,dist=self._support(coords,B)
            u=self.sym_affinity(z).transpose(1,2)
            du2=(u[:,:,:,None]-u[:,:,None,:]).pow(2)
            Sop=K*torch.exp(-0.5*du2)
            val=self.sym_value(z).view(B,S,self.n_heads,self.message_dim)
            msg=torch.einsum('bhij,bjhd->bihd',Sop,val).reshape(B,S,-1)/norm
            sym=self.sym_head(torch.cat([zt,self._gather(msg,idx)],-1)).squeeze(-1)
            if self.mode=='set_temporal_plus_sym_asym_spatial':
                q=self.direction_potential(z).transpose(1,2)
                Aop=Sop*torch.tanh(q[:,:,:,None]-q[:,:,None,:])
                aval=self.asym_value(z).view(B,S,self.n_heads,self.message_dim)
                amsg=torch.einsum('bhij,bjhd->bihd',Aop,aval).reshape(B,S,-1)/norm
                asym=self.asym_head(torch.cat([zt,self._gather(amsg,idx)],-1)).squeeze(-1)
        elif self.mode=='set_temporal_plus_unconstrained_spatial':
            zi=z[:,:,None,:].expand(B,S,S,D); zj=z[:,None,:,:].expand(B,S,S,D)
            dscale=dist/dist.max().clamp_min(1.0)
            ein=torch.cat([zi,zj,dscale[None,:,:,None].expand(B,S,S,1)],-1)
            W=torch.tanh(self.unconstrained_edge(ein)).permute(0,3,1,2)
            eye=torch.eye(S,device=z.device,dtype=z.dtype)[None,None]; W=W*(1-eye)
            val=self.unconstrained_value(z).view(B,S,self.n_heads,self.message_dim)
            msg=torch.einsum('bhij,bjhd->bihd',W,val).reshape(B,S,-1)/norm
            unc=self.unconstrained_head(torch.cat([zt,self._gather(msg,idx)],-1)).squeeze(-1)
        return sym,asym,unc,Sop,Aop
    def forward(self,x:torch.Tensor,*,target_station_index=None,coordinates=None,stochastic_gates=None):
        B,T,S,Ftr=x.shape
        if T!=self.history or S!=self.n_stations or Ftr!=self.n_features: raise ValueError('input shape mismatch')
        idx=self._indices(B,target_station_index,x.device)
        coords=self.station_coordinates if coordinates is None else torch.as_tensor(coordinates,device=x.device,dtype=x.dtype)
        z=self.station_temporal(x)
        zt=self._gather(z,idx); zmean=z.mean(1); zmax=z.max(1).values
        ctx=torch.cat([zt,zmean,zmax],-1)
        base=self.raw_head(ctx).squeeze(-1)
        sym=x.new_zeros(B); asym=x.new_zeros(B); unc=x.new_zeros(B)
        Sop=x.new_zeros(B,self.n_heads,S,S); Aop=torch.zeros_like(Sop)
        if self.mode!='set_temporal_raw_control': sym,asym,unc,Sop,Aop=self._spatial(z,idx,coords)
        aux=sym+asym+unc; corr=base+aux
        clean=torch.nan_to_num(x,nan=0.0,posinf=0.0,neginf=0.0)
        target_hist=clean.gather(2,idx[:,None,None,None].expand(B,T,1,Ftr)).squeeze(2)
        anchor=target_hist[:,-1,self.target_feature_index]
        widths=F.softplus(self.width_head(ctx))
        med=anchor+corr; q=torch.stack([med-widths[:,0],med,med+widths[:,1]],-1)
        se,ae,inner,rel=operator_errors(Sop,Aop)
        return {'quantiles':q,'latent':ctx,'median_correction':corr,'base_median_correction':base,'auxiliary_correction':aux,
                'symmetric_spatial_correction':sym,'antisymmetric_spatial_correction':asym,'unconstrained_spatial_correction':unc,
                'symmetric_operator':Sop,'antisymmetric_operator':Aop,'symmetric_operator_rms':Sop.pow(2).mean().sqrt(),
                'antisymmetric_operator_rms':Aop.pow(2).mean().sqrt(),'operator_symmetry_error':se,
                'operator_antisymmetry_error':ae,'operator_frobenius_inner_abs':inner,'operator_relative_inner_abs':rel,
                'station_embedding_rms':z.pow(2).mean().sqrt()}

def v7_parameter_count(model:nn.Module)->int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
