import torch
import torch.nn.functional as F

def weighted_heatmap_mse(logits, gt, anchor_weight=10.0, region_weight=2.0):
    pred=torch.sigmoid(logits)
    w=torch.tensor([anchor_weight,anchor_weight,region_weight], device=pred.device, dtype=pred.dtype).view(1,3,1,1)
    return (((pred-gt)**2)*w).mean()

def candidate_target_distribution(cand, target, sigma=0.05):
    d=torch.norm(cand-target[:,None,:],dim=-1)
    return torch.softmax(-d/max(float(sigma),1e-6), dim=-1).detach()

def candidate_kl_loss(probs,cand,target,sigma=0.05):
    tp=candidate_target_distribution(cand,target,sigma)
    return -(tp*torch.log(probs.clamp(min=1e-8))).sum(-1).mean()

def entropy_loss(probs):
    return -(probs*torch.log(probs.clamp(min=1e-8))).sum(-1).mean()

def anchor_faithfulness_loss(soft,a,b):
    eps=1e-6
    v=b-a; d=torch.norm(v,dim=-1,keepdim=True).clamp(min=eps)
    unit=v/d; perp=torch.stack([-unit[:,1],unit[:,0]],dim=-1)
    rel=soft-a
    alpha=(rel*unit).sum(-1,keepdim=True)/d
    beta=(rel*perp).sum(-1,keepdim=True)/d
    d1=torch.norm(soft-a,dim=-1,keepdim=True)/d
    d2=torch.norm(soft-b,dim=-1,keepdim=True)/d
    path=d1+d2
    return (F.relu(0.05-alpha).pow(2).mean() + F.relu(alpha-0.95).pow(2).mean()
            +0.35*F.relu(torch.abs(beta)-1.15).pow(2).mean()
            +0.25*F.relu(path-2.4).pow(2).mean())

def anchor_separation_loss(a,b,min_dist=0.08):
    return F.relu(min_dist-torch.norm(b-a,dim=-1)).pow(2).mean()

def schedule_lambda(epoch, pretrain_epochs=50, warmup_epochs=80, ramp_end_epoch=140):
    if epoch <= pretrain_epochs:
        return 0.0
    if epoch <= warmup_epochs:
        return 0.0
    if epoch >= ramp_end_epoch:
        return 1.0
    return float(epoch-warmup_epochs)/float(max(1,ramp_end_epoch-warmup_epochs))

def phase_name(epoch, pretrain_epochs, warmup_epochs, ramp_end_epoch):
    if epoch <= pretrain_epochs: return "anchor_pretrain"
    if epoch <= warmup_epochs: return "gt_anchor_warmup"
    if epoch < ramp_end_epoch: return "ramp_to_predicted"
    return "predicted_anchor_only"

def get_weights(args, epoch):
    pretrain = epoch <= args.anchor_pretrain_epochs
    return dict(
        heatmap=args.w_heatmap,
        anchor_heatmap_weight=args.anchor_heatmap_weight,
        region_heatmap_weight=args.region_heatmap_weight,
        anchor_coord=args.w_anchor_coord*(1.5 if epoch <= args.gt_anchor_ramp_end_epoch else 1.0),
        region_coord=args.w_region_coord,
        all_coord=args.w_all_coord,
        soft_joint=(args.w_soft_joint*0.05 if pretrain else args.w_soft_joint),
        candidate_kl=(args.w_candidate_kl*0.05 if pretrain else args.w_candidate_kl),
        anatomy=(args.w_anatomy*0.25 if pretrain else args.w_anatomy),
        anchor_separation=args.w_anchor_separation,
        entropy=args.w_entropy,
        candidate_sigma=args.candidate_sigma,
        min_anchor_dist_norm=args.min_anchor_dist_norm,
    )

def total_loss(out,batch,w):
    gt=batch["xy"]; target=gt[:,2]
    hm=weighted_heatmap_mse(out["heat_logits"], batch["heatmaps"], w["anchor_heatmap_weight"], w["region_heatmap_weight"])
    anchor=F.smooth_l1_loss(out["heat_xy"][:,:2], gt[:,:2])
    region=F.smooth_l1_loss(out["heat_xy"][:,2], target)
    allc=F.smooth_l1_loss(out["heat_xy"], gt)
    soft=F.smooth_l1_loss(out["soft_pred_xy"], target)
    kl=candidate_kl_loss(out["candidate_probs"], out["candidate_xy"], target, w["candidate_sigma"])
    anatomy=anchor_faithfulness_loss(out["soft_pred_xy"], out["path_anchor_a"], out["path_anchor_b"])
    sep=anchor_separation_loss(out["path_anchor_a"], out["path_anchor_b"], w["min_anchor_dist_norm"])
    ent=entropy_loss(out["candidate_probs"])
    loss = (w["heatmap"]*hm + w["anchor_coord"]*anchor + w["region_coord"]*region + w["all_coord"]*allc
            + w["soft_joint"]*soft + w["candidate_kl"]*kl + w["anatomy"]*anatomy
            + w["anchor_separation"]*sep + w["entropy"]*ent)
    return loss, {
        "loss_total":float(loss.detach().cpu()), "loss_heatmap":float(hm.detach().cpu()),
        "loss_anchor_coord":float(anchor.detach().cpu()), "loss_region_coord":float(region.detach().cpu()),
        "loss_all_coord":float(allc.detach().cpu()), "loss_soft_joint":float(soft.detach().cpu()),
        "loss_candidate_kl":float(kl.detach().cpu()), "loss_anatomy":float(anatomy.detach().cpu()),
        "loss_anchor_separation":float(sep.detach().cpu()), "loss_entropy":float(ent.detach().cpu()),
    }
