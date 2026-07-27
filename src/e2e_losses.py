import torch
import torch.nn.functional as F


def masked_heatmap_mse(pred_logits, target_heatmaps, mask=None):
    """
    pred_logits: [B,C,H,W]
    target_heatmaps: [B,C,H,W]
    mask: [B,C], optional
    """
    pred = torch.sigmoid(pred_logits)
    loss = (pred - target_heatmaps) ** 2

    if mask is not None:
        m = mask[:, :, None, None]
        loss = loss * m
        denom = m.sum().clamp(min=1.0) * pred.shape[-1] * pred.shape[-2]
        return loss.sum() / denom

    return loss.mean()


def select_target_channel(x, target_id):
    """
    x: [B,4,...]
    target_id: [B]
    returns [B,...]
    """
    b = x.shape[0]
    idx = torch.arange(b, device=x.device)
    return x[idx, target_id]


def entropy_loss(probs):
    ent = -(probs * (probs.clamp(min=1e-8).log())).sum(dim=-1)
    return ent.mean()


def e2e_soft_slime_loss(out, batch, weights):
    target_id = batch["target_id"]
    target_xy = batch["target_xy"]
    anchor_xy_gt = batch["anchor_xy"]
    anchor_mask = batch["anchor_mask"]

    # heatmap losses
    anchor_hm_loss = masked_heatmap_mse(
        out["anchor_logits"],
        batch["anchor_heatmaps"],
        mask=anchor_mask,
    )

    target_logits_sel = select_target_channel(out["target_logits"], target_id).unsqueeze(1)
    target_hm_sel = select_target_channel(batch["target_heatmaps"], target_id).unsqueeze(1)
    target_hm_loss = masked_heatmap_mse(target_logits_sel, target_hm_sel)

    # coord losses
    anchor_coord_diff = F.smooth_l1_loss(out["anchor_xy"], anchor_xy_gt, reduction="none").sum(dim=-1)
    anchor_coord_loss = (anchor_coord_diff * anchor_mask).sum() / anchor_mask.sum().clamp(min=1.0)

    region_xy_sel = select_target_channel(out["region_xy_all"], target_id)
    region_coord_loss = F.smooth_l1_loss(region_xy_sel, target_xy)

    joint_coord_loss = F.smooth_l1_loss(out["soft_pred_xy"], target_xy)

    ent_loss = entropy_loss(out["candidate_probs"])

    total = (
        weights["anchor_hm"] * anchor_hm_loss
        + weights["target_hm"] * target_hm_loss
        + weights["anchor_coord"] * anchor_coord_loss
        + weights["region_coord"] * region_coord_loss
        + weights["joint_coord"] * joint_coord_loss
        + weights["entropy"] * ent_loss
    )

    return total, {
        "loss_total": float(total.detach().cpu()),
        "loss_anchor_hm": float(anchor_hm_loss.detach().cpu()),
        "loss_target_hm": float(target_hm_loss.detach().cpu()),
        "loss_anchor_coord": float(anchor_coord_loss.detach().cpu()),
        "loss_region_coord": float(region_coord_loss.detach().cpu()),
        "loss_joint_coord": float(joint_coord_loss.detach().cpu()),
        "loss_entropy": float(ent_loss.detach().cpu()),
    }
