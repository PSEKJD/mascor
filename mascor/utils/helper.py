import torch
import pickle
from botorch.utils.multi_objective.pareto import is_non_dominated

# ----------------- helpers -----------------
def _normalize01(Y: torch.Tensor) -> torch.Tensor:
    ymin, ymax = Y.min(dim=0).values, Y.max(dim=0).values
    rng = torch.clamp(ymax - ymin, min=1e-12)
    return (Y - ymin) / rng

@torch.no_grad()
def min_diff_filter_on_lcox_min(pareto_obj: torch.Tensor, min_diff: float) -> torch.Tensor:
    P = pareto_obj.size(0)
    lcox = pareto_obj[:, 0]
    order = torch.argsort(lcox)
    keep = []
    for idx in order.tolist():
        if not keep:
            keep.append(idx); continue
        diffs = (lcox[idx] - lcox[torch.tensor(keep, device=lcox.device)]).abs()
        if torch.all(diffs >= min_diff):
            keep.append(idx)
    mask = torch.zeros(P, dtype=torch.bool, device=pareto_obj.device)
    mask[torch.tensor(keep, device=mask.device)] = True
    return mask

@torch.no_grad()
def fps_diverse_indices(Y: torch.Tensor, k: int = 30) -> torch.Tensor:
    N = Y.size(0)
    device = Y.device
    if N <= k:
        return torch.arange(N, device=device)
    X = _normalize01(Y.float())
    selected = []
    for m in range(X.size(1)):
        selected += [torch.argmin(X[:, m]).item(), torch.argmax(X[:, m]).item()]
    selected = list(dict.fromkeys(selected))
    sel = torch.tensor(selected, device=device, dtype=torch.long)
    min_d = torch.full((N,), float('inf'), device=device)
    if sel.numel() > 0:
        d0 = torch.cdist(X, X[sel])          # [N, |sel|]
        min_d = d0.min(1).values
        min_d[sel] = -1.0
    else:
        idx0 = torch.randint(N, (1,), device=device)
        sel = idx0
        min_d = torch.cdist(X, X[sel]).squeeze(1)
    while sel.numel() < k:
        idx = torch.argmax(min_d).item()
        sel = torch.cat([sel, torch.tensor([idx], device=device)])
        d_new = torch.cdist(X, X[idx:idx+1]).squeeze(1)
        min_d = torch.minimum(min_d, d_new)
        min_d[sel] = -1.0
    return sel

@torch.no_grad()
def min_dist_to_front_norm(dom_obj: torch.Tensor, pareto_obj: torch.Tensor) -> torch.Tensor:
    if dom_obj.numel() == 0 or pareto_obj.numel() == 0:
        return torch.zeros(dom_obj.size(0), dtype=torch.float32, device=dom_obj.device)
    Y = torch.cat([dom_obj, pareto_obj], dim=0).float()
    YN = _normalize01(Y)
    Nd = dom_obj.size(0)
    Xd, Xp = YN[:Nd], YN[Nd:]
    d = torch.cdist(Xd, Xp)            # [Nd, P]
    return d.min(1).values             # [Nd]

# ---------- main ----------
@torch.no_grad()
def select_pareto_and_dominated_min(save_path: str,
                                    country: str,
                                    region: str,
                                    min_diff: float = 0.02,
                                    dominated_k: int = 30,
                                    make_plot: bool = True,
                                    front_gap: float = 0.05,   
                                    gap_quantile: float = None):
    
    with open(save_path, "rb") as f:
        history_dict = pickle.load(f)
        
    des_list, obj_list, con_list = [], [], []
    K = len(history_dict)
    for i in range(K):
        step = history_dict[f"step-{i}"]
        des = torch.as_tensor(step["des"], dtype=torch.float32)
        if des.shape[1] == 6:
            grid_lim  = torch.as_tensor(step["grid-limit"],  dtype=torch.float32).unsqueeze(1)
            renew_lim = torch.as_tensor(step["renew-limit"], dtype=torch.float32).unsqueeze(1)
            des = torch.cat([des, grid_lim, renew_lim], dim=1)  # -> [N,8]
            pfss      = torch.as_tensor(step["pfss"],      dtype=torch.float32)
            pfss_grid = torch.as_tensor(step["pfss-grid"], dtype=torch.float32)
            con = torch.stack([pfss - 0.01, pfss_grid - 0.01], dim=1)
        else:
            pfss = torch.as_tensor(step["pfss"], dtype=torch.float32)
            con = (pfss - 0.5).unsqueeze(1)
        lcox = torch.as_tensor(step["mu-lcox[$/kg]"],     dtype=torch.float32)
        ctg  = torch.as_tensor(step["mu-ctg[ton/month]"], dtype=torch.float32)
        obj  = torch.stack([-lcox, -ctg/100], dim=1)
        
        des_list.append(des); obj_list.append(obj); con_list.append(con)

    des_set = torch.cat(des_list, dim=0)
    obj_set = torch.cat(obj_list, dim=0)
    con_set = torch.cat(con_list, dim=0)

    # 2) Feasible → Pareto / Dominated
    is_feas   = (con_set <= 0).all(-1)
    feas_des  = des_set[is_feas]
    feas_obj  = obj_set[is_feas]
    feas_con  = con_set[is_feas]
    pareto_m  = is_non_dominated(feas_obj)
    pareto_des_raw = feas_des[pareto_m]
    pareto_obj_raw = feas_obj[pareto_m]
    pareto_con_raw = feas_con[pareto_m]
    dom_des   = feas_des[~pareto_m]
    dom_obj   = feas_obj[~pareto_m]
    dom_con = feas_con[~pareto_m]

    keep_mask   = min_diff_filter_on_lcox_min(-pareto_obj_raw, min_diff)
    pareto_des  = pareto_des_raw[keep_mask]
    pareto_obj  = pareto_obj_raw[keep_mask]
    pareto_con = pareto_con_raw[keep_mask]
   
    if dom_obj.size(0) > 0 and pareto_obj.size(0) > 0:
        min_d = min_dist_to_front_norm(dom_obj, pareto_obj)  
        if front_gap is not None:
            cand_mask = min_d >= float(front_gap)
        elif gap_quantile is not None:
            thr = torch.quantile(min_d, q=float(gap_quantile))
            cand_mask = min_d >= thr
        else:
            cand_mask = torch.ones_like(min_d, dtype=torch.bool)
        cand_idx = torch.nonzero(cand_mask).squeeze(1)
       
        if cand_idx.numel() == 0:
            top = torch.argsort(min_d, descending=True)[: max(dominated_k*5, 1)]
            cand_idx = top

        sel_local = fps_diverse_indices(dom_obj[cand_idx], k=dominated_k)
        sel_idx   = cand_idx[sel_local]

        if sel_idx.numel() < dominated_k:
            picked = torch.zeros(dom_obj.size(0), dtype=torch.bool, device=dom_obj.device)
            picked[sel_idx] = True
            rest = torch.nonzero(~picked).squeeze(1)
            rest = rest[torch.argsort(min_d[rest], descending=True)]
            need = dominated_k - sel_idx.numel()
            sel_idx = torch.cat([sel_idx, rest[:need]], dim=0)

        dom_des_sel = dom_des[sel_idx]
        dom_obj_sel = dom_obj[sel_idx]
        dom_con_sel = dom_con[sel_idx]
    else:
        dom_des_sel = dom_des[:dominated_k]
        dom_obj_sel = dom_obj[:dominated_k]
        dom_con_sel = dom_con[:dominated_k]

    po = pareto_obj.detach().cpu().numpy()
    do = dom_obj_sel.detach().cpu().numpy()
    # plt.figure()
    # if po.size > 0:
    #     plt.scatter(-po[:, 0], -po[:, 1], s=22, label="Pareto (minimize)", alpha=0.9)
    # if do.size > 0:
    #     plt.scatter(-do[:, 0], -do[:, 1], s=22, marker="x",
    #                  label=f"Dominated selected (K={dominated_k})", alpha=0.9)
    # plt.xlabel("LCOX (lower is better)")
    # plt.ylabel("CTG  (lower is better)")
    # title_extra = f", gap≥{front_gap}" if front_gap is not None else (
    #      f", top≥{gap_quantile*100:.0f}%" if gap_quantile is not None else ""
    #  )
    # plt.title(f"Pareto & Dominated (diverse) {title_extra}")
    # plt.legend(); plt.savefig(f"./optimization/result/{country}/{region}/paret_curve_validation.png", dpi=600, bbox_inches='tight', transparent=True)
    # plt.show(); plt.close()

    #just checking
    dom_idx = []
    for dom_des in dom_des_sel:
        mask = torch.all(dom_des == des_set, dim=1)  
        idx = torch.nonzero(mask, as_tuple=False).squeeze()
        dom_idx.append(idx.item())
    dom_obj_error = (dom_obj_sel != obj_set[dom_idx]).float().sum()
    dom_con_error = (dom_con_sel != con_set[dom_idx]).float().sum()
    
    pareto_idx = []
    for pareto in pareto_des:
        mask = torch.all(pareto == des_set, dim=1)  
        idx = torch.nonzero(mask, as_tuple=False).squeeze()
        pareto_idx.append(idx.item())
    pareto_obj_error = (pareto_obj != obj_set[pareto_idx]).float().sum()
    pareto_con_error = (pareto_con != con_set[pareto_idx]).float().sum()
    
    return (pareto_obj, pareto_con, pareto_des, pareto_obj_error, pareto_con_error,
            dom_obj_sel, dom_con_sel, dom_des_sel, dom_obj_error, dom_con_error)