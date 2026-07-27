import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallBackbone(nn.Module):
    """
    Lightweight CNN producing 64x64 feature map for 256x256 input.
    """
    def __init__(self, out_ch=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 128

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  # 64

            nn.Conv2d(64, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


def soft_argmax_2d(logits, temperature=0.05):
    """
    logits: [B, C, H, W]
    returns:
        xy: [B, C, 2] in normalized 0..1 coords
        conf: [B, C]
    """
    b, c, h, w = logits.shape
    flat = logits.view(b, c, -1)
    prob = F.softmax(flat / max(float(temperature), 1e-6), dim=-1)

    ys = torch.linspace(0.0, 1.0, h, device=logits.device, dtype=logits.dtype)
    xs = torch.linspace(0.0, 1.0, w, device=logits.device, dtype=logits.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    x = torch.sum(prob * xx.view(1, 1, -1), dim=-1)
    y = torch.sum(prob * yy.view(1, 1, -1), dim=-1)
    xy = torch.stack([x, y], dim=-1)

    conf = torch.sigmoid(logits).view(b, c, -1).amax(dim=-1)
    return xy, conf


class SoftSlimeLayer(nn.Module):
    """
    Differentiable candidate generation + soft candidate aggregation.

    Candidate paths are generated from predicted anchors:
        anchor A -> candidate -> anchor B

    Then each candidate samples image features and receives a learned slime score.
    The final joint is the probability-weighted average of candidates.
    """
    def __init__(
        self,
        feat_dim=128,
        num_targets=4,
        alpha_steps=17,
        beta_steps=21,
        region_grid_steps=7,
        temperature=0.08,
    ):
        super().__init__()

        self.temperature = temperature

        alphas = torch.linspace(0.18, 0.82, alpha_steps)
        betas = torch.linspace(-0.95, 0.95, beta_steps)
        aa, bb = torch.meshgrid(alphas, betas, indexing="ij")
        self.register_buffer("base_alpha", aa.reshape(-1))
        self.register_buffer("base_beta", bb.reshape(-1))

        # region-local candidate offsets
        grid = torch.linspace(-1.0, 1.0, region_grid_steps)
        gx, gy = torch.meshgrid(grid, grid, indexing="ij")
        self.register_buffer("region_dx", gx.reshape(-1))
        self.register_buffer("region_dy", gy.reshape(-1))

        # target_id -> anchor indices
        # right_elbow: right_shoulder, right_wrist
        # left_elbow: left_shoulder, left_wrist
        # right_knee: right_hip, right_ankle
        # left_knee: left_hip, left_ankle
        self.register_buffer("anchor_a_table", torch.tensor([0, 4, 2, 6], dtype=torch.long))
        self.register_buffer("anchor_b_table", torch.tensor([1, 5, 3, 7], dtype=torch.long))

        geom_dim = 20 + num_targets
        self.scorer = nn.Sequential(
            nn.Linear(feat_dim + geom_dim, 160),
            nn.LayerNorm(160),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),

            nn.Linear(160, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),

            nn.Linear(128, 1),
        )

    def make_candidates(self, anchor_xy, region_xy, target_id):
        """
        anchor_xy: [B, 8, 2]
        region_xy: [B, 4, 2]
        target_id: [B]
        """
        bsz = anchor_xy.shape[0]
        device = anchor_xy.device
        dtype = anchor_xy.dtype

        a_idx = self.anchor_a_table[target_id]
        b_idx = self.anchor_b_table[target_id]

        batch_idx = torch.arange(bsz, device=device)
        a = anchor_xy[batch_idx, a_idx]  # [B,2]
        b = anchor_xy[batch_idx, b_idx]  # [B,2]
        r = region_xy[batch_idx, target_id]  # [B,2]

        v = b - a
        dist = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-4)
        unit = v / dist
        perp = torch.stack([-unit[:, 1], unit[:, 0]], dim=-1)

        alpha = self.base_alpha.to(device=device, dtype=dtype).view(1, -1, 1)
        beta = self.base_beta.to(device=device, dtype=dtype).view(1, -1, 1)

        base = a[:, None, :] + alpha * v[:, None, :]
        cand1 = base + beta * dist[:, None, :] * perp[:, None, :]

        # Region-local candidates. This is differentiable because region point is soft-argmax.
        dx = self.region_dx.to(device=device, dtype=dtype).view(1, -1, 1)
        dy = self.region_dy.to(device=device, dtype=dtype).view(1, -1, 1)
        region_radius = torch.maximum(0.18 * dist, torch.full_like(dist, 0.05))
        cand2 = r[:, None, :] + region_radius[:, None, :] * torch.cat([dx, dy], dim=-1)

        cand = torch.cat([cand1, cand2], dim=1)
        cand = cand.clamp(0.0, 1.0)

        return cand, a, b, r

    def geometry_features(self, cand, a, b, r, target_id, anchor_conf_a, anchor_conf_b):
        eps = 1e-6
        v = b - a
        d = torch.norm(v, dim=-1, keepdim=True).clamp(min=eps)
        unit = v / d
        perp = torch.stack([-unit[:, 1], unit[:, 0]], dim=-1)

        rel = cand - a[:, None, :]
        alpha = torch.sum(rel * unit[:, None, :], dim=-1, keepdim=True) / d[:, None, :]
        beta = torch.sum(rel * perp[:, None, :], dim=-1, keepdim=True) / d[:, None, :]

        d1 = torch.norm(cand - a[:, None, :], dim=-1, keepdim=True) / d[:, None, :]
        d2 = torch.norm(cand - b[:, None, :], dim=-1, keepdim=True) / d[:, None, :]
        path_ratio = d1 + d2

        v1 = a[:, None, :] - cand
        v2 = b[:, None, :] - cand
        cosang = torch.sum(v1 * v2, dim=-1, keepdim=True) / (
            torch.norm(v1, dim=-1, keepdim=True).clamp(min=eps)
            * torch.norm(v2, dim=-1, keepdim=True).clamp(min=eps)
        )
        cosang = cosang.clamp(-1.0, 1.0)

        dist_region = torch.norm(cand - r[:, None, :], dim=-1, keepdim=True) / d[:, None, :]
        region_score = torch.exp(-dist_region / 0.35)

        balance = torch.minimum(d1, d2) / (d1 + d2 + eps)

        anchor_min_conf = torch.minimum(anchor_conf_a, anchor_conf_b).view(-1, 1, 1)
        anchor_mean_conf = ((anchor_conf_a + anchor_conf_b) * 0.5).view(-1, 1, 1)

        bsz, n, _ = cand.shape
        target_onehot = F.one_hot(target_id, num_classes=4).float().view(bsz, 1, 4).expand(bsz, n, 4)

        feats = torch.cat([
            cand,
            a[:, None, :].expand_as(cand),
            b[:, None, :].expand_as(cand),
            r[:, None, :].expand_as(cand),
            alpha,
            beta,
            torch.abs(beta),
            d1,
            d2,
            path_ratio,
            cosang,
            dist_region,
            region_score,
            balance,
            anchor_min_conf.expand(bsz, n, 1),
            anchor_mean_conf.expand(bsz, n, 1),
            target_onehot,
        ], dim=-1)

        return feats

    def sample_candidate_features(self, feat_map, cand):
        """
        feat_map: [B,C,H,W]
        cand: [B,N,2] normalized 0..1
        returns: [B,N,C]
        """
        grid = cand.clone()
        grid = grid * 2.0 - 1.0
        grid = grid.view(grid.shape[0], grid.shape[1], 1, 2)
        sampled = F.grid_sample(feat_map, grid, align_corners=True)
        sampled = sampled.squeeze(-1).transpose(1, 2)
        return sampled

    def forward(self, feat_map, anchor_xy, anchor_conf, region_xy, target_id):
        cand, a, b, r = self.make_candidates(anchor_xy, region_xy, target_id)

        bsz = anchor_xy.shape[0]
        batch_idx = torch.arange(bsz, device=anchor_xy.device)
        a_idx = self.anchor_a_table[target_id]
        b_idx = self.anchor_b_table[target_id]
        conf_a = anchor_conf[batch_idx, a_idx]
        conf_b = anchor_conf[batch_idx, b_idx]

        geom = self.geometry_features(cand, a, b, r, target_id, conf_a, conf_b)
        img_feat = self.sample_candidate_features(feat_map, cand)

        score_in = torch.cat([img_feat, geom], dim=-1)
        score = self.scorer(score_in).squeeze(-1)

        probs = F.softmax(score / max(float(self.temperature), 1e-6), dim=-1)
        pred_xy = torch.sum(probs[:, :, None] * cand, dim=1)

        return {
            "candidate_xy": cand,
            "candidate_scores": score,
            "candidate_probs": probs,
            "soft_pred_xy": pred_xy,
            "path_anchor_a": a,
            "path_anchor_b": b,
            "path_region": r,
        }


class E2ESoftSlimeModel(nn.Module):
    def __init__(self, feat_dim=128, heatmap_temperature=0.05, slime_temperature=0.08):
        super().__init__()
        self.backbone = SmallBackbone(out_ch=feat_dim)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, 12, 1),
        )
        self.heatmap_temperature = heatmap_temperature
        self.slime = SoftSlimeLayer(
            feat_dim=feat_dim,
            temperature=slime_temperature,
        )

    def forward(self, image, target_id):
        feat = self.backbone(image)
        heat_logits = self.heatmap_head(feat)

        anchor_logits = heat_logits[:, :8]
        target_logits = heat_logits[:, 8:12]

        anchor_xy, anchor_conf = soft_argmax_2d(anchor_logits, temperature=self.heatmap_temperature)
        region_xy_all, region_conf_all = soft_argmax_2d(target_logits, temperature=self.heatmap_temperature)

        slime_out = self.slime(
            feat_map=feat,
            anchor_xy=anchor_xy,
            anchor_conf=anchor_conf,
            region_xy=region_xy_all,
            target_id=target_id,
        )

        return {
            "anchor_logits": anchor_logits,
            "target_logits": target_logits,
            "anchor_xy": anchor_xy,
            "anchor_conf": anchor_conf,
            "region_xy_all": region_xy_all,
            "region_conf_all": region_conf_all,
            **slime_out,
        }
