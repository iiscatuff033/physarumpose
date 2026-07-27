import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedBackbone(nn.Module):
    def __init__(self,out_ch=128):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),nn.Conv2d(32,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),nn.MaxPool2d(2),nn.Conv2d(64,out_ch,3,padding=1),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),nn.Conv2d(out_ch,out_ch,3,padding=1),nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)

def soft_argmax_2d(logits,temperature=0.05):
    b,c,h,w=logits.shape; flat=logits.reshape(b,c,-1); prob=F.softmax(flat/max(float(temperature),1e-6),dim=-1)
    ys=torch.linspace(0,1,h,device=logits.device,dtype=logits.dtype); xs=torch.linspace(0,1,w,device=logits.device,dtype=logits.dtype); yy,xx=torch.meshgrid(ys,xs,indexing='ij')
    x=torch.sum(prob*xx.reshape(-1).view(1,1,-1),dim=-1); y=torch.sum(prob*yy.reshape(-1).view(1,1,-1),dim=-1)
    return torch.stack([x,y],dim=-1), torch.sigmoid(logits).reshape(b,c,-1).amax(dim=-1)

class AnchorFaithfulSoftSlimeLayer(nn.Module):
    def __init__(self,feat_dim=128,alpha_steps=25,beta_steps=31,temperature=0.07):
        super().__init__(); self.temperature=float(temperature)
        alphas=torch.linspace(0.10,0.90,alpha_steps); betas=torch.linspace(-1.20,1.20,beta_steps); aa,bb=torch.meshgrid(alphas,betas,indexing='ij')
        self.register_buffer('base_alpha',aa.reshape(-1)); self.register_buffer('base_beta',bb.reshape(-1))
        geom_dim=27
        self.scorer=nn.Sequential(nn.Linear(feat_dim+geom_dim,192),nn.LayerNorm(192),nn.ReLU(inplace=True),nn.Dropout(0.10),nn.Linear(192,128),nn.LayerNorm(128),nn.ReLU(inplace=True),nn.Dropout(0.10),nn.Linear(128,1))
    def make_anchor_path_candidates(self,a,b):
        v=b-a; dist=torch.norm(v,dim=-1,keepdim=True).clamp(min=1e-4); unit=v/dist; perp=torch.stack([-unit[:,1],unit[:,0]],dim=-1)
        alpha=self.base_alpha.to(device=a.device,dtype=a.dtype).view(1,-1,1); beta=self.base_beta.to(device=a.device,dtype=a.dtype).view(1,-1,1)
        return (a[:,None,:]+alpha*v[:,None,:]+beta*dist[:,None,:]*perp[:,None,:]).clamp(0,1)
    def sample_features(self,feat_map,cand):
        grid=(cand*2-1).view(cand.shape[0],cand.shape[1],1,2); sampled=F.grid_sample(feat_map,grid,align_corners=True)
        return sampled.squeeze(-1).transpose(1,2)
    def geometry_features(self,cand,a,b,r,target_id,heat_conf):
        eps=1e-6; v=b-a; d=torch.norm(v,dim=-1,keepdim=True).clamp(min=eps); unit=v/d; perp=torch.stack([-unit[:,1],unit[:,0]],dim=-1)
        rel=cand-a[:,None,:]; alpha=torch.sum(rel*unit[:,None,:],dim=-1,keepdim=True)/d[:,None,:]; beta=torch.sum(rel*perp[:,None,:],dim=-1,keepdim=True)/d[:,None,:]
        d1=torch.norm(cand-a[:,None,:],dim=-1,keepdim=True)/d[:,None,:]; d2=torch.norm(cand-b[:,None,:],dim=-1,keepdim=True)/d[:,None,:]; path_ratio=d1+d2
        v1=a[:,None,:]-cand; v2=b[:,None,:]-cand; cosang=(torch.sum(v1*v2,dim=-1,keepdim=True)/(torch.norm(v1,dim=-1,keepdim=True).clamp(min=eps)*torch.norm(v2,dim=-1,keepdim=True).clamp(min=eps))).clamp(-1,1)
        dist_region=torch.norm(cand-r[:,None,:],dim=-1,keepdim=True)/d[:,None,:]; region_score=torch.exp(-dist_region/0.35); balance=torch.minimum(d1,d2)/(d1+d2+eps)
        bsz,n,_=cand.shape; conf_a=heat_conf[:,0].view(bsz,1,1); conf_b=heat_conf[:,1].view(bsz,1,1); conf_r=heat_conf[:,2].view(bsz,1,1); target_onehot=F.one_hot(target_id,num_classes=4).float().view(bsz,1,4).expand(bsz,n,4)
        return torch.cat([cand,a[:,None,:].expand_as(cand),b[:,None,:].expand_as(cand),r[:,None,:].expand_as(cand),alpha,beta,torch.abs(beta),d1,d2,path_ratio,cosang,dist_region,region_score,balance,conf_a.expand(bsz,n,1),conf_b.expand(bsz,n,1),conf_r.expand(bsz,n,1),torch.minimum(conf_a,conf_b).expand(bsz,n,1),((conf_a+conf_b)*0.5).expand(bsz,n,1),target_onehot],dim=-1)
    def forward(self,feat_map,heat_xy,heat_conf,target_id):
        a=heat_xy[:,0]; b=heat_xy[:,1]; r=heat_xy[:,2]; cand=self.make_anchor_path_candidates(a,b); geom=self.geometry_features(cand,a,b,r,target_id,heat_conf); img_feat=self.sample_features(feat_map,cand)
        scores=self.scorer(torch.cat([img_feat,geom],dim=-1)).squeeze(-1); probs=F.softmax(scores/max(self.temperature,1e-6),dim=-1); soft_pred=torch.sum(probs[:,:,None]*cand,dim=1)
        return {'candidate_xy':cand,'candidate_scores':scores,'candidate_probs':probs,'soft_pred_xy':soft_pred,'path_anchor_a':a,'path_anchor_b':b,'path_region':r}

class AnchorFaithfulPhysarumPoseModel(nn.Module):
    def __init__(self,feat_dim=128,target_embed_dim=16,heatmap_temperature=0.05,slime_temperature=0.07):
        super().__init__(); self.backbone=SharedBackbone(out_ch=feat_dim); self.target_embed=nn.Embedding(4,target_embed_dim)
        self.heatmap_head=nn.Sequential(nn.Conv2d(feat_dim+target_embed_dim,feat_dim,3,padding=1),nn.BatchNorm2d(feat_dim),nn.ReLU(inplace=True),nn.Conv2d(feat_dim,feat_dim,3,padding=1),nn.BatchNorm2d(feat_dim),nn.ReLU(inplace=True),nn.Conv2d(feat_dim,3,1))
        self.heatmap_temperature=float(heatmap_temperature); self.slime=AnchorFaithfulSoftSlimeLayer(feat_dim=feat_dim,temperature=slime_temperature)
    def forward(self,image,target_id):
        feat=self.backbone(image); b,c,h,w=feat.shape; emb=self.target_embed(target_id); emb_map=emb[:,:,None,None].expand(b,emb.shape[1],h,w); heat_logits=self.heatmap_head(torch.cat([feat,emb_map],dim=1)); heat_xy,heat_conf=soft_argmax_2d(heat_logits,temperature=self.heatmap_temperature)
        slime_out=self.slime(feat,heat_xy,heat_conf,target_id)
        return {'heat_logits':heat_logits,'heat_xy':heat_xy,'heat_conf':heat_conf,**slime_out}
