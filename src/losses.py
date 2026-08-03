import torch
import torch.nn.functional as F

def masked_heatmap_mse(alog,rlog,hm,mask,aw=20.0,rw=1.4):
    pred=torch.cat([torch.sigmoid(alog),torch.sigmoid(rlog)],1); w=torch.cat([torch.full((8,),aw,device=pred.device,dtype=pred.dtype),torch.full((4,),rw,device=pred.device,dtype=pred.dtype)]).view(1,12,1,1); m=mask.view(mask.shape[0],12,1,1); return (((pred-hm)**2)*w*m).sum()/(w*m).sum().clamp(min=1.0)
def masked_coord(pred,gt,mask):
    loss=F.smooth_l1_loss(pred,gt,reduction='none').sum(-1)*mask; return loss.sum()/mask.sum().clamp(min=1.0)
def selected_pair_loss(anchor_xy,gt_pair,pair):
    B=anchor_xy.shape[0]; idx=torch.arange(B,device=anchor_xy.device); pred=torch.stack([anchor_xy[idx,pair[:,0]],anchor_xy[idx,pair[:,1]]],1); return F.smooth_l1_loss(pred,gt_pair)
def selected_target_xy(region_xy,tid):
    idx=torch.arange(region_xy.shape[0],device=region_xy.device); return region_xy[idx,tid]
def cand_dist(cand,target,sigma=.045):
    return torch.softmax(-torch.norm(cand-target[:,None,:],dim=-1)/max(float(sigma),1e-6),dim=-1).detach()
def kl_loss(probs,cand,target,sigma=.045):
    tp=cand_dist(cand,target,sigma); return -(tp*torch.log(probs.clamp(min=1e-8))).sum(-1).mean()
def entropy(probs): return -(probs*torch.log(probs.clamp(min=1e-8))).sum(-1).mean()
def anatomy(soft,a,b):
    eps=1e-6; v=b-a; d=torch.norm(v,dim=-1,keepdim=True).clamp(min=eps); u=v/d; p=torch.stack([-u[:,1],u[:,0]],-1); rel=soft-a; alpha=(rel*u).sum(-1,keepdim=True)/d; beta=(rel*p).sum(-1,keepdim=True)/d; d1=torch.norm(soft-a,dim=-1,keepdim=True)/d; d2=torch.norm(soft-b,dim=-1,keepdim=True)/d
    return F.relu(.05-alpha).pow(2).mean()+F.relu(alpha-.95).pow(2).mean()+.35*F.relu(torch.abs(beta)-1.15).pow(2).mean()+.25*F.relu(d1+d2-2.4).pow(2).mean()
def sep_loss(a,b,min_d=.08): return F.relu(min_d-torch.norm(b-a,dim=-1)).pow(2).mean()
def offset_reg(aoff,roff): return (aoff**2).mean()+.5*(roff**2).mean()
def schedule_lambda(epoch,pretrain_epochs=100,warmup_epochs=150,ramp_end_epoch=240):
    if epoch<=pretrain_epochs or epoch<=warmup_epochs: return 0.0
    if epoch>=ramp_end_epoch: return 1.0
    return float(epoch-warmup_epochs)/float(max(1,ramp_end_epoch-warmup_epochs))
def phase_name(epoch,pretrain,warmup,ramp):
    if epoch<=pretrain: return 'semantic_anchor_offset_pretrain'
    if epoch<=warmup: return 'gt_semantic_anchor_warmup'
    if epoch<ramp: return 'ramp_to_predicted_semantic_anchors'
    return 'predicted_semantic_anchor_only'
def get_weights(args,epoch):
    pre=epoch<=args.anchor_pretrain_epochs
    return {'heatmap':args.w_heatmap,'anchor_heatmap_weight':args.anchor_heatmap_weight,'region_heatmap_weight':args.region_heatmap_weight,'all_anchor_coord':args.w_all_anchor_coord,'raw_anchor_coord':args.w_raw_anchor_coord,'selected_anchor_coord':args.w_selected_anchor_coord*(1.25 if epoch<=args.gt_anchor_ramp_end_epoch else 1.0),'region_coord':args.w_region_coord,'soft_joint':args.w_soft_joint*(.05 if pre else 1.0),'candidate_kl':args.w_candidate_kl*(.05 if pre else 1.0),'anatomy':args.w_anatomy*(.25 if pre else 1.0),'anchor_separation':args.w_anchor_separation,'offset_reg':args.w_offset_reg,'entropy':args.w_entropy,'candidate_sigma':args.candidate_sigma,'min_anchor_dist_norm':args.min_anchor_dist_norm}
def total_loss(out,batch,pair,gt_pair,target,w):
    hm=masked_heatmap_mse(out['anchor_logits'],out['region_logits'],batch['heatmaps'],batch['heatmap_mask'],w['anchor_heatmap_weight'],w['region_heatmap_weight']); all_a=masked_coord(out['anchor_xy'],batch['anchor_xy'],batch['anchor_mask']); raw=masked_coord(out['anchor_xy_raw'],batch['anchor_xy'],batch['anchor_mask']); sel=selected_pair_loss(out['anchor_xy'],gt_pair,pair); reg=masked_coord(out['region_xy'],batch['region_xy'],batch['region_mask']); soft=F.smooth_l1_loss(out['soft_pred_xy'],target); kl=kl_loss(out['candidate_probs'],out['candidate_xy'],target,w['candidate_sigma']); an=anatomy(out['soft_pred_xy'],out['path_anchor_a'],out['path_anchor_b']); sep=sep_loss(out['path_anchor_a'],out['path_anchor_b'],w['min_anchor_dist_norm']); off=offset_reg(out['anchor_offsets'],out['region_offsets']); ent=entropy(out['candidate_probs'])
    loss=w['heatmap']*hm+w['all_anchor_coord']*all_a+w['raw_anchor_coord']*raw+w['selected_anchor_coord']*sel+w['region_coord']*reg+w['soft_joint']*soft+w['candidate_kl']*kl+w['anatomy']*an+w['anchor_separation']*sep+w['offset_reg']*off+w['entropy']*ent
    return loss,{k:float(v.detach().cpu()) for k,v in {'loss_total':loss,'loss_heatmap':hm,'loss_all_anchor_coord':all_a,'loss_raw_anchor_coord':raw,'loss_selected_anchor_coord':sel,'loss_region_coord':reg,'loss_soft_joint':soft,'loss_candidate_kl':kl,'loss_anatomy':an,'loss_anchor_separation':sep,'loss_offset_reg':off,'loss_entropy':ent}.items()}
