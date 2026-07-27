import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedBackbone(nn.Module):
    """
    Lightweight 256 -> 64 feature map CNN.
    Replace with HRNet/ViT later if needed.
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
    logits: [B,C,H,W]
    returns xy [B,C,2] normalized 0..1 crop coords, conf [B,C]
    """
    b, c, h, w = logits.shape
    flat = logits.reshape(b, c, -1)
    prob = F.softmax(flat / max(float(temperature), 1e-6), dim=-1)

    ys = torch.linspace(0.0, 1.0, h, device=logits.device, dtype=logits.dtype)
    xs = torch.linspace(0.0, 1.0, w, device=logits.device, dtype=logits.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    x = torch.sum(prob * xx.view(1, 1, -1), dim=-1)
    y = torch.sum(prob * yy.view(1, 1, -1), dim=-1)
    xy = torch.stack([x, y], dim=-1)

    conf = torch.sigmoid(logits).reshape(b, c, -1).amax(dim=-1)

    return xy, conf


class DifferentiableSoftSlimeLayer(nn.Module):
    """
    Fully differentiable candidate generation and soft reinforcement.

    Inputs:
        feature map
        predicted anchor A
        predicted anchor B
        predicted region point
        target id

    Output:
        soft final joint coordinate
        candidate probabilities
        candidate coordinates
    """
    def __init__(
        self,
        feat_dim=128,
        target_embed_dim=16,
        alpha_steps=19,
        beta_steps=25,
        region_grid_steps=7,
        temperature=0.07,
    ):
        super().__init__()

        self.temperature = float(temperature)

        alphas = torch.linspace(0.16, 0.84, alpha_steps)
        betas = torch.linspace(-1.05, 1.05, beta_steps)
        aa, bb = torch.meshgrid(alphas, betas, indexing="ij")
        self.register_buffer("base_alpha", aa.reshape(-1))
        self.register_buffer("base_beta", bb.reshape(-1))

        grid = torch.linspace(-1.0, 1.0, region_grid_steps)
        gx, gy = torch.meshgrid(grid, grid, indexing="ij")
        self.register_buffer("region_dx", gx.reshape(-1))
        self.register_buffer("region_dy", gy.reshape(-1))

        # Geometry feature count is 27: 23 scalar/vector geometry/confidence features + 4 target one-hot features.
        geom_dim = 23 + 4
        self.scorer = nn.Sequential(
            nn.Linear(feat_dim + geom_dim, 192),
            nn.LayerNorm(192),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),

            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.10),

            nn.Linear(128, 1),
        )

    def make_candidates(self, a, b, r):
        bsz = a.shape[0]
        device = a.device
        dtype = a.dtype

        v = b - a
        dist = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-4)
        unit = v / dist
        perp = torch.stack([-unit[:, 1], unit[:, 0]], dim=-1)

        alpha = self.base_alpha.to(device=device, dtype=dtype).view(1, -1, 1)
        beta = self.base_beta.to(device=device, dtype=dtype).view(1, -1, 1)

        base = a[:, None, :] + alpha * v[:, None, :]
        cand_path = base + beta * dist[:, None, :] * perp[:, None, :]

        dx = self.region_dx.to(device=device, dtype=dtype).view(1, -1, 1)
        dy = self.region_dy.to(device=device, dtype=dtype).view(1, -1, 1)
        radius = torch.maximum(0.18 * dist, torch.full_like(dist, 0.05))
        cand_region = r[:, None, :] + radius[:, None, :] * torch.cat([dx, dy], dim=-1)

        cand = torch.cat([cand_path, cand_region], dim=1).clamp(0.0, 1.0)
        return cand

    def geometry_features(self, cand, a, b, r, target_id, heat_conf):
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

        bsz, n, _ = cand.shape

        conf_a = heat_conf[:, 0].view(bsz, 1, 1)
        conf_b = heat_conf[:, 1].view(bsz, 1, 1)
        conf_r = heat_conf[:, 2].view(bsz, 1, 1)

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
            conf_a.expand(bsz, n, 1),
            conf_b.expand(bsz, n, 1),
            conf_r.expand(bsz, n, 1),
            torch.minimum(conf_a, conf_b).expand(bsz, n, 1),
            ((conf_a + conf_b) * 0.5).expand(bsz, n, 1),
            target_onehot,
        ], dim=-1)

        return feats

    def sample_features(self, feat_map, cand):
        grid = cand * 2.0 - 1.0
        grid = grid.view(cand.shape[0], cand.shape[1], 1, 2)
        sampled = F.grid_sample(feat_map, grid, align_corners=True)
        sampled = sampled.squeeze(-1).transpose(1, 2)
        return sampled

    def forward(self, feat_map, heat_xy, heat_conf, target_id):
        a = heat_xy[:, 0]
        b = heat_xy[:, 1]
        r = heat_xy[:, 2]

        cand = self.make_candidates(a, b, r)
        geom = self.geometry_features(cand, a, b, r, target_id, heat_conf)
        img_feat = self.sample_features(feat_map, cand)

        score_input = torch.cat([img_feat, geom], dim=-1)
        scores = self.scorer(score_input).squeeze(-1)

        probs = F.softmax(scores / max(self.temperature, 1e-6), dim=-1)
        soft_pred = torch.sum(probs[:, :, None] * cand, dim=1)

        return {
            "candidate_xy": cand,
            "candidate_scores": scores,
            "candidate_probs": probs,
            "soft_pred_xy": soft_pred,
            "path_anchor_a": a,
            "path_anchor_b": b,
            "path_region": r,
        }


class UnifiedPhysarumPoseModel(nn.Module):
    """
    Single end-to-end differentiable model.

    Input:
        crop image
        target joint type id

    Output:
        anchor A/B heatmaps,
        region heatmap,
        differentiable slime soft joint,
        top-K path hypotheses at eval time.
    """
    def __init__(
        self,
        feat_dim=128,
        target_embed_dim=16,
        heatmap_temperature=0.05,
        slime_temperature=0.07,
    ):
        super().__init__()

        self.backbone = UnifiedBackbone(out_ch=feat_dim)
        self.target_embed = nn.Embedding(4, target_embed_dim)

        self.heatmap_head = nn.Sequential(
            nn.Conv2d(feat_dim + target_embed_dim, feat_dim, 3, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(feat_dim, feat_dim, 3, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(feat_dim, 3, 1),
        )

        self.heatmap_temperature = float(heatmap_temperature)
        self.slime = DifferentiableSoftSlimeLayer(
            feat_dim=feat_dim,
            target_embed_dim=target_embed_dim,
            temperature=slime_temperature,
        )

    def forward(self, image, target_id):
        feat = self.backbone(image)
        b, c, h, w = feat.shape

        emb = self.target_embed(target_id)
        emb_map = emb[:, :, None, None].expand(b, emb.shape[1], h, w)

        cond_feat = torch.cat([feat, emb_map], dim=1)
        heat_logits = self.heatmap_head(cond_feat)

        heat_xy, heat_conf = soft_argmax_2d(
            heat_logits,
            temperature=self.heatmap_temperature,
        )

        slime_out = self.slime(
            feat_map=feat,
            heat_xy=heat_xy,
            heat_conf=heat_conf,
            target_id=target_id,
        )

        return {
            "heat_logits": heat_logits,
            "heat_xy": heat_xy,
            "heat_conf": heat_conf,
            **slime_out,
        }
