import torch
import torch.nn.functional as F

def weighted_heatmap_mse(pred_logits,gt_heatmaps,anchor_weight=8.0,region_weight=2.0):
    pred=torch.sigmoid(pred_logits); weights=torch.tensor([anchor_weight,anchor_weight,region_weight],dtype=pred.dtype,device=pred.device).view(1,3,1,1)
    return (((pred-gt_heatmaps)**2)*weights).mean()

def candidate_target_distribution(candidate_xy,target_xy,sigma=0.05):
    d=torch.norm(candidate_xy-target_xy[:,None,:],dim=-1); return torch.softmax(-d/max(float(sigma),1e-6),dim=-1).detach()

def candidate_kl_loss(pred_probs,candidate_xy,target_xy,sigma=0.05):
    target_probs=candidate_target_distribution(candidate_xy,target_xy,sigma=sigma)
    return -(target_probs*torch.log(pred_probs.clamp(min=1e-8))).sum(dim=-1).mean()

def entropy_loss(probs): return -(probs*torch.log(probs.clamp(min=1e-8))).sum(dim=-1).mean()

def anchor_faithfulness_loss(soft_pred,anchor_a,anchor_b):
    eps=1e-6; v=anchor_b-anchor_a; d=torch.norm(v,dim=-1,keepdim=True).clamp(min=eps); unit=v/d; perp=torch.stack([-unit[:,1],unit[:,0]],dim=-1)
    rel=soft_pred-anchor_a; alpha=torch.sum(rel*unit,dim=-1,keepdim=True)/d; beta=torch.sum(rel*perp,dim=-1,keepdim=True)/d
    d1=torch.norm(soft_pred-anchor_a,dim=-1,keepdim=True)/d; d2=torch.norm(soft_pred-anchor_b,dim=-1,keepdim=True)/d; path_ratio=d1+d2
    return F.relu(0.05-alpha).pow(2).mean()+F.relu(alpha-0.95).pow(2).mean()+0.35*F.relu(torch.abs(beta)-1.15).pow(2).mean()+0.25*F.relu(path_ratio-2.4).pow(2).mean()

def anchor_separation_loss(anchor_a,anchor_b,min_dist=0.08): return F.relu(min_dist-torch.norm(anchor_b-anchor_a,dim=-1)).pow(2).mean()

def anchor_faithful_loss(out,batch,weights):
    gt_xy=batch['xy']; target_xy=gt_xy[:,2]
    hm=weighted_heatmap_mse(out['heat_logits'],batch['heatmaps'],weights['anchor_heatmap_weight'],weights['region_heatmap_weight'])
    anchor_coord=F.smooth_l1_loss(out['heat_xy'][:,:2],gt_xy[:,:2]); region_coord=F.smooth_l1_loss(out['heat_xy'][:,2],target_xy); all_coord=F.smooth_l1_loss(out['heat_xy'],gt_xy); soft_joint=F.smooth_l1_loss(out['soft_pred_xy'],target_xy)
    cand_kl=candidate_kl_loss(out['candidate_probs'],out['candidate_xy'],target_xy,sigma=weights['candidate_sigma']); anatomy=anchor_faithfulness_loss(out['soft_pred_xy'],out['path_anchor_a'],out['path_anchor_b']); separation=anchor_separation_loss(out['path_anchor_a'],out['path_anchor_b'],weights['min_anchor_dist_norm']); ent=entropy_loss(out['candidate_probs'])
    total=weights['heatmap']*hm+weights['anchor_coord']*anchor_coord+weights['region_coord']*region_coord+weights['all_coord']*all_coord+weights['soft_joint']*soft_joint+weights['candidate_kl']*cand_kl+weights['anatomy']*anatomy+weights['anchor_separation']*separation+weights['entropy']*ent
    return total,{'loss_total':float(total.detach().cpu()),'loss_heatmap':float(hm.detach().cpu()),'loss_anchor_coord':float(anchor_coord.detach().cpu()),'loss_region_coord':float(region_coord.detach().cpu()),'loss_all_coord':float(all_coord.detach().cpu()),'loss_soft_joint':float(soft_joint.detach().cpu()),'loss_candidate_kl':float(cand_kl.detach().cpu()),'loss_anatomy':float(anatomy.detach().cpu()),'loss_anchor_separation':float(separation.detach().cpu()),'loss_entropy':float(ent.detach().cpu())}

def get_curriculum_weights(args,epoch):
    base={'heatmap':args.w_heatmap,'anchor_heatmap_weight':args.anchor_heatmap_weight,'region_heatmap_weight':args.region_heatmap_weight,'anchor_coord':args.w_anchor_coord,'region_coord':args.w_region_coord,'all_coord':args.w_all_coord,'soft_joint':args.w_soft_joint,'candidate_kl':args.w_candidate_kl,'anatomy':args.w_anatomy,'anchor_separation':args.w_anchor_separation,'entropy':args.w_entropy,'candidate_sigma':args.candidate_sigma,'min_anchor_dist_norm':args.min_anchor_dist_norm}
    if epoch<=args.anchor_warmup_epochs:
        base['anchor_coord']*=1.6; base['soft_joint']*=0.35; base['candidate_kl']*=0.25
    return base
