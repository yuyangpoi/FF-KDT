import torch



def event_EST(events, height, width, start_times, durations, nbins, mode='bilinear', vol=None):
    """
    Densifies events into an volume
    (uniform cut)

    Args:
        events (tensor): N,4 (x,y,p,t)
        height (int): height of output volume
        width (int): width of output volume
        start_times: (B,) start times of each volume
        durations: (B,) durations for each volume
        nbins (int): number of time bins for output volume
        mode (str): either "bilinear" or "nearest" interpolation of voxels.
    """
    xs = events[:, -4].long()
    ys = events[:, -3].long()
    ps = events[:, -2].float()
    ts = events[:, -1].float()

    ti_star = (ts - start_times) * (nbins-1) / durations
    lbin = torch.floor(ti_star)
    lbin = torch.clamp(lbin, min=0, max=nbins - 2)
    if vol is None:
        vol = torch.zeros((nbins*2, height, width), dtype=torch.float32, device=events.device)
    if mode == 'bilinear':
        rbin = torch.clamp(lbin + 1, max=nbins - 1)
        lvals = torch.clamp(1 - torch.abs(lbin - ti_star), min=0)
        rvals = 1 - lvals


        pos_mask = ps > 0
        neg_mask = ~pos_mask
        ## positive
        vol.index_put_((lbin.long()[pos_mask]*2, ys[pos_mask], xs[pos_mask]), lvals[pos_mask], accumulate=True)
        vol.index_put_((rbin.long()[pos_mask]*2, ys[pos_mask], xs[pos_mask]), rvals[pos_mask], accumulate=True)
        ## negative
        vol.index_put_((lbin.long()[neg_mask]*2+1, ys[neg_mask], xs[neg_mask]), lvals[neg_mask], accumulate=True)
        vol.index_put_((rbin.long()[neg_mask]*2+1, ys[neg_mask], xs[neg_mask]), rvals[neg_mask], accumulate=True)
    else:
        raise NotImplementedError
    return vol  # [C, H, W]











