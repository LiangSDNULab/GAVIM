import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, view_dim, latent_dim):
        super(Encoder, self).__init__()
        self.x_dim = view_dim
        self.z_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(self.x_dim, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 256),
                                     nn.ReLU()
                                     )
        self.z_mu = nn.Linear(256, self.z_dim)
        self.z_var = nn.Sequential(nn.Linear(256, self.z_dim), nn.Softplus())

    def forward(self, x):
        hidden_feature = self.encoder(x)
        vs_mu = self.z_mu(hidden_feature)
        vs_var = self.z_var(hidden_feature)
        return vs_mu, vs_var
    

class Decoder(nn.Module):
    def __init__(self, view_dim, latent_dim):
        super(Decoder, self).__init__()
        self.x_dim = view_dim
        self.z_dim = latent_dim
        self.decoder = nn.Sequential(nn.Linear(self.z_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 512),
                                     nn.ReLU(),
                                     nn.Linear(512, self.x_dim),
                                     )

    def forward(self, z):
        xr = self.decoder(z)
        return xr


class Gaussian_sampling(nn.Module):
    def forward(self, mu, var):
        std = torch.sqrt(var)
        epi = std.data.new(std.size()).normal_()
        return epi * std + mu


class Gaussian_WB(nn.Module):
    def forward(self, mu, var, mask=None):
        if mask is None:
            mask_matrix = torch.ones(mu.shape[0], mu.shape[1], 1).to(mu.device)
        else:
            mask_matrix = torch.stack(mask, dim=0)
        std = torch.sqrt(var)
        exist_mu = mu * mask_matrix
        exist_std = std * mask_matrix
        aggregate_mu = torch.sum(exist_mu, dim=0) / torch.sum(mask_matrix, dim=0)
        aggregate_std  = torch.sum(exist_std, dim=0) / torch.sum(mask_matrix, dim=0)
        return aggregate_mu, aggregate_std**2
    

class GAVIM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.x_dim_list = args.multiview_dims
        self.z_dim = args.z_dim
        self.num_views = args.num_views

        self.encoders = nn.ModuleDict({f'view_{v}': Encoder(self.x_dim_list[v], self.z_dim) for v in range(self.num_views)})
        self.decoders = nn.ModuleDict({f'view_{v}': Decoder(self.x_dim_list[v], self.z_dim) for v in range(self.num_views)})

        self.aggregated_fn = Gaussian_WB()
        self.sampling_fn = Gaussian_sampling()

    def generation_x(self, z):
        xr_list = [vs_decoder(z) for vs_decoder in self.decoders.values()]
        return xr_list

    def inference(self, x_list, mask):
        vs_mus, vs_vars, vs_sample = [], [], []
        for v in range(self.num_views):
            vs_mu, vs_var = self.encoders[f'view_{v}'](x_list[v])
            vs_mus.append(vs_mu)
            vs_vars.append(vs_var)
            vs_sample.append(self.sampling_fn(vs_mu, vs_var))
        mu = torch.stack(vs_mus)
        var = torch.stack(vs_vars)
        aggregated_mu, aggregated_var = self.aggregated_fn(mu, var, mask)
        return vs_sample, aggregated_mu, aggregated_var

    def forward(self, x_list, mask=None):
        vs_sample, aggregated_mu, aggregated_var = self.inference(x_list, mask)
        z_sample = self.sampling_fn(aggregated_mu, aggregated_var)
        xr_list = self.generation_x(z_sample)
        return z_sample, vs_sample, aggregated_mu, aggregated_var, xr_list
    
    def inference_z(self, x_list, mask):
        vs_mus, vs_vars = [], []
        for v in range(self.num_views):
            vs_mu, vs_var = self.encoders[f'view_{v}'](x_list[v])
            vs_mus.append(vs_mu)
            vs_vars.append(vs_var)
        mu = torch.stack(vs_mus)
        var = torch.stack(vs_vars)
        aggregated_mu, aggregated_var = self.aggregated_fn(mu, var, mask)
        return aggregated_mu, aggregated_var
    
    def sv_encode(self, x, v):
        v_mu, v_var = self.encoders[f'view_{v}'](x)
        v_sample = self.sampling_fn(v_mu, v_var)
        return v_sample

