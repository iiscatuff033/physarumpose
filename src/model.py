import torch
import torch.nn as nn
import torch.nn.functional as F

class Backbone(nn.Module):
    def __init__(self, out_ch=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64,out_ch,3,padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch,out_ch,3,padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
        )
    def forward(self,x): return self.net(x)

def soft_argmax_2d(logits, temperature=0.05):
    b,c,h,w = logits.shape
    flat = logits.reshape(b,c,-1)
    prob = F.softmax(flat/max(float(temperature),1e-6), dim=-1)
    ys = torch.linspace(0,1,h,device=logits.device,dtype=logits.dtype)
    xs = torch.linspace(0,1,w,device=logits.device,dtype=logits.dtype)
    yy,xx = torch.meshgrid(ys,xs,indexing="ij")
    x = (prob*xx.reshape(-1).view(1,1,-1)).sum(-1)
    y = (prob*yy.reshape(-1).view(1,1,-1)).sum(-1)
    conf = torch.sigmoid(logits).reshape(b,c,-1).amax(-1)
    return torch.stack([x,y], dim=-1), conf

class ScheduledSlime(nn.Module):
    def __init__(self, feat_dim=128, alpha_steps=25, beta_steps=31, temperature=0.07):
        super().__init__()
        self.temperature = float(temperature)
        alphas = torch.linspace(0.10,0.90,alpha_steps)
        betas = torch.linspace(-1.20,1.20,beta_steps)
        aa,bb = torch.meshgrid(alphas,betas,indexing="ij")
        self.register_buffer("base_alpha", aa.reshape(-1))
        self.register_buffer("base_beta", bb.reshape(-1))
        geom_dim = 27
        self.scorer = nn.Sequential(
            nn.Linear(feat_dim+geom_dim,192), nn.LayerNorm(192), nn.ReLU(True), nn.Dropout(0.10),
            nn.Linear(192,128), nn.LayerNorm(128), nn.ReLU(True), nn.Dropout(0.10),
            nn.Linear(128,1),
        )

    def make_candidates(self,a,b):
        v = b-a
        d = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-4)
        unit = v/d
        perp = torch.stack([-unit[:,1], unit[:,0]], dim=-1)
        alpha = self.base_alpha.to(a.device,a.dtype).view(1,-1,1)
        beta = self.base_beta.to(a.device,a.dtype).view(1,-1,1)
        cand = a[:,None,:] + alpha*v[:,None,:] + beta*d[:,None,:]*perp[:,None,:]
        return cand.clamp(0,1)

    def sample_feat(self, feat, cand):
        grid = cand*2-1
        grid = grid.view(cand.shape[0], cand.shape[1], 1, 2)
        s = F.grid_sample(feat, grid, align_corners=True)
        return s.squeeze(-1).transpose(1,2)

    def geom(self,cand,a,b,r,target_id,conf):
        eps=1e-6
        v=b-a; d=torch.norm(v,dim=-1,keepdim=True).clamp(min=eps)
        unit=v/d; perp=torch.stack([-unit[:,1],unit[:,0]],dim=-1)
        rel=cand-a[:,None,:]
        alpha=(rel*unit[:,None,:]).sum(-1,keepdim=True)/d[:,None,:]
        beta=(rel*perp[:,None,:]).sum(-1,keepdim=True)/d[:,None,:]
        d1=torch.norm(cand-a[:,None,:],dim=-1,keepdim=True)/d[:,None,:]
        d2=torch.norm(cand-b[:,None,:],dim=-1,keepdim=True)/d[:,None,:]
        path_ratio=d1+d2
        v1=a[:,None,:]-cand; v2=b[:,None,:]-cand
        cosang=(v1*v2).sum(-1,keepdim=True)/(torch.norm(v1,dim=-1,keepdim=True).clamp(min=eps)*torch.norm(v2,dim=-1,keepdim=True).clamp(min=eps))
        cosang=cosang.clamp(-1,1)
        dist_region=torch.norm(cand-r[:,None,:],dim=-1,keepdim=True)/d[:,None,:]
        region_score=torch.exp(-dist_region/0.35)
        balance=torch.minimum(d1,d2)/(d1+d2+eps)
        bsz,n,_=cand.shape
        ca=conf[:,0].view(bsz,1,1); cb=conf[:,1].view(bsz,1,1); cr=conf[:,2].view(bsz,1,1)
        onehot=F.one_hot(target_id,num_classes=4).float().view(bsz,1,4).expand(bsz,n,4)
        return torch.cat([
            cand, a[:,None,:].expand_as(cand), b[:,None,:].expand_as(cand), r[:,None,:].expand_as(cand),
            alpha,beta,torch.abs(beta),d1,d2,path_ratio,cosang,dist_region,region_score,balance,
            ca.expand(bsz,n,1), cb.expand(bsz,n,1), cr.expand(bsz,n,1),
            torch.minimum(ca,cb).expand(bsz,n,1), ((ca+cb)*0.5).expand(bsz,n,1),
            onehot
        ], dim=-1)

    def forward(self, feat, pred_xy, conf, target_id, gt_xy=None, gt_anchor_lambda=1.0):
        pred_a,pred_b,r = pred_xy[:,0],pred_xy[:,1],pred_xy[:,2]
        if gt_xy is not None:
            lam=float(gt_anchor_lambda)
            a = lam*pred_a + (1-lam)*gt_xy[:,0]
            b = lam*pred_b + (1-lam)*gt_xy[:,1]
        else:
            a,b=pred_a,pred_b
        cand=self.make_candidates(a,b)
        inp=torch.cat([self.sample_feat(feat,cand), self.geom(cand,a,b,r,target_id,conf)], dim=-1)
        scores=self.scorer(inp).squeeze(-1)
        probs=F.softmax(scores/max(self.temperature,1e-6), dim=-1)
        soft=(probs[:,:,None]*cand).sum(1)
        return dict(candidate_xy=cand,candidate_scores=scores,candidate_probs=probs,soft_pred_xy=soft,
                    path_anchor_a=a,path_anchor_b=b,path_region=r,pred_anchor_a=pred_a,pred_anchor_b=pred_b)

class PhysarumPoseR1(nn.Module):
    def __init__(self, feat_dim=128, embed_dim=16, heatmap_temperature=0.05, slime_temperature=0.07):
        super().__init__()
        self.backbone=Backbone(feat_dim)
        self.embed=nn.Embedding(4,embed_dim)
        self.head=nn.Sequential(
            nn.Conv2d(feat_dim+embed_dim,feat_dim,3,padding=1), nn.BatchNorm2d(feat_dim), nn.ReLU(True),
            nn.Conv2d(feat_dim,feat_dim,3,padding=1), nn.BatchNorm2d(feat_dim), nn.ReLU(True),
            nn.Conv2d(feat_dim,3,1),
        )
        self.heatmap_temperature=float(heatmap_temperature)
        self.slime=ScheduledSlime(feat_dim=feat_dim, temperature=slime_temperature)

    def forward(self,image,target_id,gt_xy=None,gt_anchor_lambda=1.0):
        feat=self.backbone(image)
        b,c,h,w=feat.shape
        emb=self.embed(target_id)[:,:,None,None].expand(b,-1,h,w)
        logits=self.head(torch.cat([feat,emb],dim=1))
        xy,conf=soft_argmax_2d(logits,self.heatmap_temperature)
        out=self.slime(feat,xy,conf,target_id,gt_xy=gt_xy,gt_anchor_lambda=gt_anchor_lambda)
        return dict(heat_logits=logits,heat_xy=xy,heat_conf=conf,**out)
