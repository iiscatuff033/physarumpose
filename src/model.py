import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self,ic,oc,stride=1):
        super().__init__(); self.c1=nn.Conv2d(ic,oc,3,stride,padding=1,bias=False); self.b1=nn.BatchNorm2d(oc); self.c2=nn.Conv2d(oc,oc,3,padding=1,bias=False); self.b2=nn.BatchNorm2d(oc)
        self.skip=nn.Identity() if ic==oc and stride==1 else nn.Sequential(nn.Conv2d(ic,oc,1,stride,bias=False),nn.BatchNorm2d(oc))
    def forward(self,x):
        y=F.relu(self.b1(self.c1(x)),inplace=True); y=self.b2(self.c2(y)); return F.relu(y+self.skip(x),inplace=True)
class StrongBackbone(nn.Module):
    def __init__(self,out_ch=160):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(3,48,3,padding=1,bias=False),nn.BatchNorm2d(48),nn.ReLU(True),nn.Conv2d(48,48,3,padding=1,bias=False),nn.BatchNorm2d(48),nn.ReLU(True),ResidualBlock(48,80,2),ResidualBlock(80,80),ResidualBlock(80,128,2),ResidualBlock(128,128),ResidualBlock(128,128),nn.Conv2d(128,out_ch,3,padding=1,bias=False),nn.BatchNorm2d(out_ch),nn.ReLU(True),ResidualBlock(out_ch,out_ch))
    def forward(self,x): return self.net(x)
def soft_argmax_2d(logits,temp=0.04):
    b,c,h,w=logits.shape; prob=F.softmax(logits.reshape(b,c,-1)/max(float(temp),1e-6),dim=-1)
    ys=torch.linspace(0,1,h,device=logits.device,dtype=logits.dtype); xs=torch.linspace(0,1,w,device=logits.device,dtype=logits.dtype); yy,xx=torch.meshgrid(ys,xs,indexing='ij')
    x=(prob*xx.reshape(-1).view(1,1,-1)).sum(-1); y=(prob*yy.reshape(-1).view(1,1,-1)).sum(-1); conf=torch.sigmoid(logits).reshape(b,c,-1).amax(-1)
    return torch.stack([x,y],-1),conf
def sample_offsets(offmap,xy):
    b,two,h,w=offmap.shape; c=xy.shape[1]; off=offmap.view(b,c,2,h,w); outs=[]
    for j in range(c):
        grid=xy[:,j].view(b,1,1,2)*2-1; val=F.grid_sample(off[:,j],grid,align_corners=True).squeeze(-1).squeeze(-1); outs.append(val)
    return torch.stack(outs,1)
class Slime(nn.Module):
    def __init__(self,feat_dim=160,alpha_steps=27,beta_steps=33,temp=0.06):
        super().__init__(); self.temp=float(temp); a=torch.linspace(0.08,0.92,alpha_steps); b=torch.linspace(-1.25,1.25,beta_steps); aa,bb=torch.meshgrid(a,b,indexing='ij'); self.register_buffer('ba',aa.reshape(-1)); self.register_buffer('bb',bb.reshape(-1)); self.scorer=nn.Sequential(nn.Linear(feat_dim+27,224),nn.LayerNorm(224),nn.ReLU(True),nn.Dropout(.1),nn.Linear(224,160),nn.LayerNorm(160),nn.ReLU(True),nn.Dropout(.1),nn.Linear(160,1))
    def candidates(self,a,b):
        v=b-a; d=torch.norm(v,dim=-1,keepdim=True).clamp(min=1e-4); u=v/d; p=torch.stack([-u[:,1],u[:,0]],-1); A=self.ba.to(a.device,a.dtype).view(1,-1,1); B=self.bb.to(a.device,a.dtype).view(1,-1,1); return (a[:,None,:]+A*v[:,None,:]+B*d[:,None,:]*p[:,None,:]).clamp(0,1)
    def sample_feat(self,feat,cand):
        grid=cand*2-1; grid=grid.view(cand.shape[0],cand.shape[1],1,2); return F.grid_sample(feat,grid,align_corners=True).squeeze(-1).transpose(1,2)
    def geom(self,cand,a,b,r,tid,ca,cb,cr):
        eps=1e-6; v=b-a; d=torch.norm(v,dim=-1,keepdim=True).clamp(min=eps); u=v/d; p=torch.stack([-u[:,1],u[:,0]],-1); rel=cand-a[:,None,:]
        alpha=(rel*u[:,None,:]).sum(-1,keepdim=True)/d[:,None,:]; beta=(rel*p[:,None,:]).sum(-1,keepdim=True)/d[:,None,:]
        d1=torch.norm(cand-a[:,None,:],dim=-1,keepdim=True)/d[:,None,:]; d2=torch.norm(cand-b[:,None,:],dim=-1,keepdim=True)/d[:,None,:]
        v1=a[:,None,:]-cand; v2=b[:,None,:]-cand; cos=(v1*v2).sum(-1,keepdim=True)/(torch.norm(v1,dim=-1,keepdim=True).clamp(min=eps)*torch.norm(v2,dim=-1,keepdim=True).clamp(min=eps))
        dr=torch.norm(cand-r[:,None,:],dim=-1,keepdim=True)/d[:,None,:]; rs=torch.exp(-dr/.35); bal=torch.minimum(d1,d2)/(d1+d2+eps); Bsz,N,_=cand.shape; ca=ca.view(Bsz,1,1); cb=cb.view(Bsz,1,1); cr=cr.view(Bsz,1,1); oh=F.one_hot(tid,num_classes=4).float().view(Bsz,1,4).expand(Bsz,N,4)
        return torch.cat([cand,a[:,None,:].expand_as(cand),b[:,None,:].expand_as(cand),r[:,None,:].expand_as(cand),alpha,beta,torch.abs(beta),d1,d2,d1+d2,cos.clamp(-1,1),dr,rs,bal,ca.expand(Bsz,N,1),cb.expand(Bsz,N,1),cr.expand(Bsz,N,1),torch.minimum(ca,cb).expand(Bsz,N,1),((ca+cb)*.5).expand(Bsz,N,1),oh],-1)
    def forward(self,feat,anchor_xy,anchor_conf,region_xy,region_conf,tid,pair,gt_pair=None,lam=1.0):
        B=anchor_xy.shape[0]; idx=torch.arange(B,device=anchor_xy.device); ai=pair[:,0]; bi=pair[:,1]; pa=anchor_xy[idx,ai]; pb=anchor_xy[idx,bi]; ca=anchor_conf[idx,ai]; cb=anchor_conf[idx,bi]; r=region_xy[idx,tid]; cr=region_conf[idx,tid]
        if gt_pair is not None:
            a=float(lam)*pa+(1-float(lam))*gt_pair[:,0]; b=float(lam)*pb+(1-float(lam))*gt_pair[:,1]
        else: a,b=pa,pb
        cand=self.candidates(a,b); inp=torch.cat([self.sample_feat(feat,cand),self.geom(cand,a,b,r,tid,ca,cb,cr)],-1); scores=self.scorer(inp).squeeze(-1); probs=F.softmax(scores/max(self.temp,1e-6),dim=-1); soft=(probs[:,:,None]*cand).sum(1)
        return {'candidate_xy':cand,'candidate_scores':scores,'candidate_probs':probs,'soft_pred_xy':soft,'path_anchor_a':a,'path_anchor_b':b,'path_region':r}
class Stage6ESingleModel(nn.Module):
    def __init__(self,feat_dim=160,embed_dim=24,heatmap_temp=0.04,slime_temp=0.06,max_anchor_offset=.035,max_region_offset=.035):
        super().__init__(); self.backbone=StrongBackbone(feat_dim); self.embed=nn.Embedding(4,embed_dim); self.heatmap_temp=heatmap_temp; self.max_anchor_offset=max_anchor_offset; self.max_region_offset=max_region_offset
        def head(out): return nn.Sequential(nn.Conv2d(feat_dim+embed_dim,feat_dim,3,padding=1),nn.BatchNorm2d(feat_dim),nn.ReLU(True),nn.Conv2d(feat_dim,feat_dim,3,padding=1),nn.BatchNorm2d(feat_dim),nn.ReLU(True),nn.Conv2d(feat_dim,out,1))
        self.anchor_h=head(8); self.anchor_o=head(16); self.region_h=head(4); self.region_o=head(8); self.slime=Slime(feat_dim,temp=slime_temp)
    def forward(self,img,tid,pair_indices,gt_anchor_pair=None,gt_anchor_lambda=1.0):
        feat=self.backbone(img); B,C,H,W=feat.shape; emb=self.embed(tid)[:,:,None,None].expand(B,-1,H,W); cond=torch.cat([feat,emb],1)
        alog=self.anchor_h(cond); aomap=self.anchor_o(cond); rlog=self.region_h(cond); romap=self.region_o(cond)
        araw,aconf=soft_argmax_2d(alog,self.heatmap_temp); rraw,rconf=soft_argmax_2d(rlog,self.heatmap_temp)
        aoff=torch.tanh(sample_offsets(aomap,araw))*self.max_anchor_offset; roff=torch.tanh(sample_offsets(romap,rraw))*self.max_region_offset
        axy=(araw+aoff).clamp(0,1); rxy=(rraw+roff).clamp(0,1)
        out=self.slime(feat,axy,aconf,rxy,rconf,tid,pair_indices,gt_anchor_pair,gt_anchor_lambda)
        return {'anchor_logits':alog,'region_logits':rlog,'anchor_xy_raw':araw,'region_xy_raw':rraw,'anchor_offsets':aoff,'region_offsets':roff,'anchor_xy':axy,'region_xy':rxy,'anchor_conf':aconf,'region_conf':rconf,**out}
