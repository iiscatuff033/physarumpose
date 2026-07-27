import torch
import torch.nn.functional as F


def heatmap_mse(pred_logits, gt_heatmaps):
    pred = torch.sigmoid(pred_logits)
    return ((pred - gt_heatmaps) ** 2).mean()


def candidate_target_distribution(candidate_xy, target_xy, sigma=0.06):
    d = torch.norm(candidate_xy - target_xy[:, None, :], dim=-1)
    logits = -d / max(float(sigma), 1e-6)
    return torch.softmax(logits, dim=-1).detach()


def candidate_kl_loss(pred_probs, candidate_xy, target_xy, sigma=0.06):
    target_probs = candidate_target_distribution(candidate_xy, target_xy, sigma=sigma)
    return -(target_probs * torch.log(pred_probs.clamp(min=1e-8))).sum(dim=-1).mean()


def entropy_loss(probs):
    return -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1).mean()


def unified_physarum_loss(out, batch, weights):
    gt_xy = batch["xy"]
    target_xy = gt_xy[:, 2]

    hm_loss = heatmap_mse(out["heat_logits"], batch["heatmaps"])

    coord_loss_all = F.smooth_l1_loss(out["heat_xy"], gt_xy)
    anchor_coord_loss = F.smooth_l1_loss(out["heat_xy"][:, :2], gt_xy[:, :2])
    region_coord_loss = F.smooth_l1_loss(out["heat_xy"][:, 2], target_xy)

    soft_joint_loss = F.smooth_l1_loss(out["soft_pred_xy"], target_xy)

    cand_kl = candidate_kl_loss(
        out["candidate_probs"],
        out["candidate_xy"],
        target_xy,
        sigma=weights.get("candidate_sigma", 0.06),
    )

    ent = entropy_loss(out["candidate_probs"])

    total = (
        weights["heatmap"] * hm_loss
        + weights["coord_all"] * coord_loss_all
        + weights["anchor_coord"] * anchor_coord_loss
        + weights["region_coord"] * region_coord_loss
        + weights["soft_joint"] * soft_joint_loss
        + weights["candidate_kl"] * cand_kl
        + weights["entropy"] * ent
    )

    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_heatmap": float(hm_loss.detach().cpu()),
        "loss_coord_all": float(coord_loss_all.detach().cpu()),
        "loss_anchor_coord": float(anchor_coord_loss.detach().cpu()),
        "loss_region_coord": float(region_coord_loss.detach().cpu()),
        "loss_soft_joint": float(soft_joint_loss.detach().cpu()),
        "loss_candidate_kl": float(cand_kl.detach().cpu()),
        "loss_entropy": float(ent.detach().cpu()),
    }
