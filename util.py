import torch

def get_tensor_shape(x):
    a = list(x.shape)
    b = [x.size(i) for i in range(len(a))]
    def _select_one(aa, bb):
        if isinstance(aa, int):
            return aa
        else:
            return bb
    return [_select_one(aa, bb) for aa, bb in zip(a, b)]

def pass_net_nx(func, in_img, n):
    b, h, w, c = get_tensor_shape(in_img)
    def _get_nx(x):
        s, r = divmod(x, n)
        s = s if r == 0 else s + 1
        return n*s
    nx_h = _get_nx(h)
    nx_w = _get_nx(w)
    def _get_rl_rr(x, nx):
        r = nx - x
        rl = r//2
        rr = r - rl
        return rl, rr
    padding = (0, 0, _get_rl_rr(h, nx_h)[0], _get_rl_rr(h, nx_h)[1], _get_rl_rr(w, nx_w)[0], _get_rl_rr(w, nx_w)[1])
    in_img = torch.nn.functional.pad(in_img, padding, mode='reflect')
    in_img = in_img.view(b, nx_h, nx_w, c)
    out_img = func(in_img)
    out_img = torch.nn.functional.pad(out_img, (-_get_rl_rr(h, nx_h)[0], -_get_rl_rr(h, nx_h)[1], -_get_rl_rr(w, nx_w)[0], -_get_rl_rr(w, nx_w)[1]))
    return out_img

def sample_1d(img, y_idx): # img: [b, h, c]; y_idx: [b, n]
    b, h, c = get_tensor_shape(img)
    b, n = get_tensor_shape(y_idx)
    b_idx = torch.arange(b).view(-1, 1).expand(-1, n)
    y_idx = torch.clamp(y_idx, 0, h - 1)
    a_idx = torch.stack([b_idx, y_idx], dim=-1)
    return img[a_idx[...,0], a_idx[...,1]]

def interp_1d(img, y): # img: [b, h, c]; y: [b, n]
    b, h, c = get_tensor_shape(img)
    b, n = get_tensor_shape(y)
    y_0 = torch.floor(y)
    y_1 = y_0 + 1
    y_0_val = sample_1d(img, y_0.int())
    y_1_val = sample_1d(img, y_1.int())
    w_0 = y_1 - y
    w_1 = y - y_0
    w_0 = w_0.unsqueeze(-1)
    w_1 = w_1.unsqueeze(-1)
    return w_0 * y_0_val + w_1 * y_1_val

def apply_rf(x, rf):
    b, *s = get_tensor_shape(x)
    b, k = get_tensor_shape(rf)
    x = interp_1d(
        rf.unsqueeze(-1),
        ((k - 1) * x.reshape(b, -1)).float()
    )
    return x.reshape(b, *s)

def get_l2_loss(a, b):
    return torch.mean((a - b)**2)

def get_l2_loss_with_mask(a, b):
    return torch.mean((a - b)**2, dim=[1, 2, 3], keepdim=True)

def quantize(img, s=255):
    _clip = lambda x: torch.clamp(x, 0, 1)
    img = _clip(img)
    img = torch.round(s * img) / s
    img = _clip(img)
    return img

def rand_quantize(img, is_training):
    b, h, w, c = get_tensor_shape(img)
    rand_bit = torch.randint(8, 12, (b, 1, 1, 1)).float()
    const_bit = torch.tensor(8.0).expand_as(rand_bit)
    bit = torch.where(is_training, rand_bit, const_bit)
    s = (2**bit) - 1
    return quantize(img, s)