import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from e3nn import o3
    try:
        from e3nn.nn import Linear as E3Linear
    except Exception:
        try:
            from e3nn.o3 import Linear as E3Linear
        except Exception:
            E3Linear = None
except Exception:
    o3 = None
    E3Linear = None

def gather_nodes(nodes, neighbor_idx):
    """
    Gather node features at neighbor indices.
    nodes: [B, N, C]
    neighbor_idx: [B, N, K]
    returns: [B, N, K, C]
    """
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    return neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])


class EquiFormerBlock(nn.Module):
    """
    Lightweight EquiFormer-style block using e3nn Linear to produce
    scalar and vector channels from scalar node features, then performing
    neighbor attention that is scalar-valued while transporting vector channels.

    Note: This is a simplified, practical PoC for integrating e3nn into
    the existing codebase. It does not implement the full tensor-product
    based EquiFormer but provides a clear interface to replace the encoder.
    """
    def __init__(
        self,
        in_scalar_dim,
        out_scalar_dim,
        out_vector_dim,
        hidden_att=128,
        dropout=0.1,
        in_vector_dim=0,
        edge_feat_dim=0,
        edge_att_dim=0,
    ):
        """
        in_scalar_dim: number of input scalar features per node
        out_scalar_dim: number of output scalar features per node
        out_vector_dim: number of output vector channels (each is 3D)
        in_vector_dim: number of input vector channels (optional); if >0, forward accepts vector_in [B,N,Cv,3]
        """
        super(EquiFormerBlock, self).__init__()
        self.in_scalar_dim = in_scalar_dim
        self.in_vector_dim = in_vector_dim
        self.out_scalar_dim = out_scalar_dim
        self.out_vector_dim = out_vector_dim
        self.dropout = nn.Dropout(dropout)
        self.edge_feat_dim = edge_feat_dim
        self.edge_att_dim = edge_att_dim

        if edge_feat_dim and edge_att_dim:
            self.edge_proj = nn.Linear(edge_feat_dim, edge_att_dim)
            att_in_dim = out_scalar_dim * 2 + 1 + edge_att_dim
        else:
            self.edge_proj = None
            att_in_dim = out_scalar_dim * 2 + 1

        # e3nn Linear: input can be scalar-only or scalar+vector; output is scalar+vector
        # in_irreps: (in_scalar x 0e) or (in_scalar x 0e + in_vector x 1e)
        # out_irreps: (out_scalar x 0e + out_vector x 1e)
        if (E3Linear is not None) and (o3 is not None):
            if in_vector_dim > 0:
                in_irreps = o3.Irreps(f"{in_scalar_dim}x0e + {in_vector_dim}x1e")
            else:
                in_irreps = o3.Irreps(f"{in_scalar_dim}x0e")
            out_irreps = o3.Irreps(f"{out_scalar_dim}x0e + {out_vector_dim}x1e")
            self.e3_lin = E3Linear(in_irreps, out_irreps)
            self._e3nn_available = True
        else:
            in_dim = in_scalar_dim + in_vector_dim * 3
            out_dim_fallback = out_scalar_dim + out_vector_dim * 3
            self.e3_lin = nn.Linear(in_dim, out_dim_fallback)
            self._e3nn_available = False

        # Attention MLP (central + neighbor + distance [+ edge])
        self.att_mlp = nn.Sequential(
            nn.Linear(att_in_dim, hidden_att),
            nn.GELU(),
            nn.Linear(hidden_att, 1)
        )

        # Final scalar FFN
        self.scalar_ffn = nn.Sequential(
            nn.Linear(out_scalar_dim, out_scalar_dim * 4),
            nn.GELU(),
            nn.Linear(out_scalar_dim * 4, out_scalar_dim)
        )
        self.scalar_norm = nn.LayerNorm(out_scalar_dim)

    def forward(self, scalar_feats, coords, E_idx, mask=None, vector_in=None, edge_feats=None):
        """
        scalar_feats: [B, N, Cs]  (float)
        coords: [B, N, 3]
        E_idx: [B, N, K] neighbor indices
        mask: [B, N] binary mask (optional)
        vector_in: [B, N, Cv_in, 3] optional input vector features (e.g. backbone frame)

        returns: scalar_out [B,N,out_scalar_dim], vector_out [B,N,out_vector_dim,3]
        """
        B, N, Cs = scalar_feats.shape
        K = E_idx.shape[-1]
        device = scalar_feats.device

        # Build linear input: scalar + optional vector (flattened)
        if self.in_vector_dim > 0 and vector_in is not None:
            # vector_in [B, N, Cv, 3] -> [B, N, Cv*3]
            lin_input = torch.cat([scalar_feats, vector_in.reshape(B, N, -1)], dim=-1)
        else:
            lin_input = scalar_feats

        # Linear map to mixed irreps and split into scalar and vector parts
        if getattr(self, "_e3nn_available", False):
            lin_out = self.e3_lin(lin_input)  # e3nn Linear -> [B,N, out_scalar + 3*out_vector]
        else:
            lin_out = self.e3_lin(lin_input)  # fallback nn.Linear -> [B,N,out_dim]
        total_out_dim = lin_out.shape[-1]
        scalar_dim = self.out_scalar_dim
        vec_dim = self.out_vector_dim
        scalar_out = lin_out[..., :scalar_dim]  # [B,N,scalar_dim]
        vec_flat = lin_out[..., scalar_dim:]  # [B,N, 3*vec_dim] or empty
        if vec_dim > 0:
            vec_out = vec_flat.view(B, N, vec_dim, 3)  # [B,N,vec_dim,3]
        else:
            vec_out = torch.zeros(B, N, 0, 3, device=device)

        # Gather neighbor scalars and coords
        neigh_scalar = gather_nodes(scalar_out, E_idx)  # [B,N,K,scalar_dim]
        neigh_coords = gather_nodes(coords, E_idx)  # [B,N,K,3]
        central_coords = coords.unsqueeze(-2).expand(-1, -1, K, -1)  # [B,N,K,3]

        # Relative vectors and distances
        r_ij = central_coords - neigh_coords  # [B,N,K,3]
        d_ij = torch.sqrt((r_ij ** 2).sum(-1) + 1e-6).unsqueeze(-1)  # [B,N,K,1]
        u_ij = r_ij / (d_ij + 1e-6)  # unit vectors

        # Attention scores from central scalar, neighbor scalar and distance (+ optional edge feats)
        central_scalar = scalar_out.unsqueeze(-2).expand(-1, -1, K, -1)  # [B,N,K,scalar_dim]
        att_parts = [
            central_scalar.reshape(B, N, K, -1),
            neigh_scalar.reshape(B, N, K, -1),
            d_ij
        ]
        if self.edge_proj is not None and edge_feats is not None:
            att_parts.append(self.edge_proj(edge_feats))
        att_input = torch.cat(att_parts, dim=-1)

        att_raw = self.att_mlp(att_input)  # [B,N,K,1]
        if mask is not None:
            neigh_mask = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)  # [B,N,K]
            att_raw = att_raw.masked_fill((neigh_mask == 0).unsqueeze(-1), -1e4)
        # 混合精度下防止 float16 溢出：限制 logits 范围，避免 softmax 前 exp 爆炸
        att_raw = att_raw.clamp(min=-1e4, max=10.0)

        att = F.softmax(att_raw, dim=2)  # softmax over K neighbors

        # Message passing: aggregate neighbor scalar messages
        # Reduce neighbor scalar (projected via linear_out)
        neigh_scalar_proj = neigh_scalar  # already in projected scalar space
        scalar_msg = (att * neigh_scalar_proj).sum(dim=2)  # [B,N,scalar_dim]

        # Aggregate vector channels by transporting neighbor vector channels along unit vectors
        # Create neighbor vector features: gather vec_out
        if vec_dim > 0:
            neigh_vec = gather_nodes(vec_out.view(B, N, vec_dim * 3), E_idx)  # [B,N,K,vec_dim*3]
            neigh_vec = neigh_vec.view(B, N, K, vec_dim, 3)  # [B,N,K,vec_dim,3]
            # Option A: use neigh_vec directly weighted by attention
            vec_msg = (att.unsqueeze(-1) * neigh_vec).sum(dim=2)  # [B,N,vec_dim,3]
        else:
            vec_msg = torch.zeros(B, N, 0, 3, device=device)

        # Update scalar features with residual + FFN
        scalar_out = self.scalar_norm(scalar_out + self.dropout(scalar_msg))
        scalar_out = scalar_out + self.dropout(self.scalar_ffn(scalar_out))

        return scalar_out, vec_msg

