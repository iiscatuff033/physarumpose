import torch
import torch.nn.functional as F


def heatmap_mse(pred_logits, target_heatmaps):
    pred = torch.sigmoid(pred_logits)
    return ((pred - target_heatmaps) ** 2).mean()


def select_target_channel(x, target_id):
    b = x.shape[0]
    idx = torch.arange(b, device=x.device)
    return x[idx, target_id]


def entropy_loss(probs):
    ent = -(probs * (probs.clamp(min=1e-8).log())).sum(dim=-1)
    return ent.mean()


def target_candidate_distribution(candidate_xy, target_xy, sigma=0.06):
    # candidate_xy: [B,N,2], target_xy: [B,2]
    d = torch.norm(candidate_xy - target_xy[:, None, :], dim=-1)
    logits = -d / max(float(sigma), 1e-6)
    return torch.softmax(logits, dim=-1).detach()


def kl_candidate_loss(pred_probs, candidate_xy, target_xy, sigma=0.06):
    target_probs = target_candidate_distribution(candidate_xy, target_xy, sigma=sigma)
    logp = torch.log(pred_probs.clamp(min=1e-8))
    return -(target_probs * logp).sum(dim=-1).mean()


def teacher_anchor_soft_slime_loss(out, batch, weights):
    target_id = batch["target_id"]
    target_xy = batch["target_xy"]

    target_logits_sel = select_target_channel(out["target_logits"], target_id).unsqueeze(1)
    target_heat_sel = select_target_channel(batch["target_heatmaps"], target_id).unsqueeze(1)

    target_hm_loss = heatmap_mse(target_logits_sel, target_heat_sel)

    region_xy_sel = select_target_channel(out["region_xy_all"], target_id)
    region_coord_loss = F.smooth_l1_loss(region_xy_sel, target_xy)

    joint_coord_loss = F.smooth_l1_loss(out["soft_pred_xy"], target_xy)

    candidate_kl = kl_candidate_loss(
        out["candidate_probs"],
        out["candidate_xy"],
        target_xy,
        sigma=weights.get("candidate_sigma", 0.06),
    )

    ent_loss = entropy_loss(out["candidate_probs"])

    total = (
        weights["target_hm"] * target_hm_loss
        + weights["region_coord"] * region_coord_loss
        + weights["joint_coord"] * joint_coord_loss
        + weights["candidate_kl"] * candidate_kl
        + weights["entropy"] * ent_loss
    )

    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_target_hm": float(target_hm_loss.detach().cpu()),
        "loss_region_coord": float(region_coord_loss.detach().cpu()),
        "loss_joint_coord": float(joint_coord_loss.detach().cpu()),
        "loss_candidate_kl": float(candidate_kl.detach().cpu()),
        "loss_entropy": float(ent_loss.detach().cpu()),
    }
