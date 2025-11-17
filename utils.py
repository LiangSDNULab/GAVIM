import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).sum(2)
    return torch.exp(-kernel_input)


def compute_maximum_mean_discrepancy(x, y=None):
    if y is None:
        y = torch.randn_like(x)
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd


def gaussian_kl(q_mu, q_var, p_mu=None, p_var=None): 
    if p_mu is None:
        p_mu = torch.zeros_like(q_mu)
    if p_var is None:
        p_var = torch.ones_like(q_var)
    kl = - 0.5 * (torch.log(q_var / p_var) - q_var / p_var - torch.pow(q_mu - p_mu, 2) / p_var + 1)
    return kl.sum(-1).mean()


class SimKernel(nn.Module):
    """Enhanced Similarity computation kernel with optimizations."""
    VALID_METRICS = {'euclidean', 'cosine'}
    
    def __init__(self, metric: str = 'euclidean', k_neighbor=3, p=2):
        super().__init__()
        self.metric = metric
        self.k_neighbor = k_neighbor
        self.p = p

    def forward(self, features, k_features=None, k_mask=None, knn=True) -> torch.Tensor:
        # Handle dictionary-based input automatically
        if k_features is None:
            Similaritys = self._compute(features)
            Similaritys.fill_diagonal_(float(0))

            if knn:
                # Get KNN indices
                _, nn_idx = torch.topk(Similaritys, k=min(self.k_neighbor, Similaritys.size(1)), dim=1, largest=True)
                
                # Create adjacency matrix
                n = Similaritys.size(0)
                adj_matrix = torch.zeros_like(Similaritys)
                row_idx = torch.arange(n, device=Similaritys.device).view(-1, 1).expand(-1, self.k_neighbor)
                adj_matrix[row_idx.flatten(), nn_idx.flatten()] = 1.0
            else:
                deg = Similaritys.sum(dim=1, keepdim=True)
                adj_matrix = Similaritys / (deg + 1e-10)

        else:
            Similaritys = self._compute_k(features, k_features)
            # Similaritys = Similaritys * k_mask
            
            deg = Similaritys.sum(dim=1, keepdim=True)
            adj_matrix = Similaritys / (deg + 1e-10)

        return adj_matrix
            
    def _compute(self, features):
        """Compute pairwise Similaritys with optimizations."""
        # Process input
        x1 = x2 = features
        
        # Compute Similaritys based on metric
        if self.metric == 'euclidean':
            return self._euclidean_Similarity(x1, x2)
        elif self.metric == 'cosine':
            return self._cosine_Similarity(x1, x2)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _euclidean_Similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Optimized Euclidean Similarity computation."""
        x1_norm = (x1**2).sum(1).view(-1, 1)
        x2_norm = (x2**2).sum(1).view(1, -1)
        dist = x1_norm + x2_norm - 2.0 * torch.mm(x1, x2.t())
        return torch.exp(-torch.clamp(dist, 0.0, float('inf')))

    def _cosine_Similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Optimized cosine Similarity computation."""
        x1_normalized = F.normalize(x1, p=self.p, dim=1)
        x2_normalized = F.normalize(x2, p=self.p, dim=1)
        return torch.exp(torch.mm(x1_normalized, x2_normalized.t()))
    
    def _compute_k(self, features, k_features):
        """Compute pairwise Similaritys with optimizations."""
        features = features.unsqueeze(1).expand(-1, k_features.shape[1], -1)

        # Compute Similaritys based on metric
        if self.metric == 'euclidean':
            return self._euclidean_Similarity_k(features, k_features)
        elif self.metric == 'cosine':
            return self._cosine_Similarity_k(features, k_features)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _euclidean_Similarity_k(self, features: torch.Tensor, k_features: torch.Tensor) -> torch.Tensor:
        """Optimized Euclidean Similarity computation."""
        distances = torch.norm(features - k_features, dim=2) ** 2
        return torch.exp(-torch.clamp(distances, 0.0, float('inf')))

    def _cosine_Similarity_k(self, features: torch.Tensor, k_features: torch.Tensor) -> torch.Tensor:
        """Optimized cosine Similarity computation."""
        x1_normalized = F.normalize(features, p=self.p, dim=2)
        x2_normalized = F.normalize(k_features, p=self.p, dim=2)
        similarity = F.cosine_similarity(x1_normalized, x2_normalized, dim=2)
        return torch.exp(similarity)



def Smooth_P(adj_matrix, k):
    n = adj_matrix.shape[0]
    P_tilde = torch.zeros_like(adj_matrix)
    P_power = torch.eye(n, device=adj_matrix.device)
    for _ in range(1, k + 1):
        P_power = P_power @ adj_matrix
        P_tilde += P_power
    
    P_tilde.fill_diagonal_(float(0))

    deg = P_tilde.sum(dim=1, keepdim=True)
    P_tilde = P_tilde / (deg + 1e-10)
    return P_tilde


def volume_computation(anchor, inputs, masks=None):
    """
    General function to compute volume for contrastive learning loss functions.
    Compute the volume metric for each vector in anchor batch and all the other modalities listed in *inputs.

    Args:
    - anchor (torch.Tensor): Tensor of shape (batch_size1, dim)
    - inputs (torch.Tensor): Variable number of tensors of shape (batch_size2, dim)

    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """
    anchor = F.normalize(anchor, p=2, dim=1)
    inputs = [F.normalize(input, p=2, dim=1) for input in inputs]

    batch_size1 = anchor.shape[0]
    batch_size2 = inputs[0].shape[0]

    # Compute pairwise dot products for language with itself
    aa = torch.einsum('bi,bi->b', anchor, anchor).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with each input
    l_inputs = [anchor @ input.T for input in inputs]

    # Compute pairwise dot products for each input with themselves and with each other
    input_dot_products = []
    for i, input1 in enumerate(inputs):
        row = []
        for j, input2 in enumerate(inputs):
            dot_product = torch.einsum('bi,bi->b', input1, input2).unsqueeze(0).expand(batch_size1, -1)
            row.append(dot_product)
        input_dot_products.append(row)

    # Stack the results to form the Gram matrix for each pair
    G = torch.stack([
        torch.stack([aa] + l_inputs, dim=-1),
        *[torch.stack([l_inputs[i]] + input_dot_products[i], dim=-1) for i in range(len(inputs))]
    ], dim=-2)

    if masks is None:
        gram_det = torch.det(G.float())

    else:
        mask = torch.cat([torch.ones((batch_size1, 1)).to(masks[0].device)] + masks, dim=1)
        gram_det = []
        for j in range(batch_size2):
            mat = G[:, j, :, :]
            mas = mask[j, :].bool()
            trimmed = mat[:, mas, :][:, :, mas]
            det = torch.det(trimmed.float())
            gram_det.append(det)
        gram_det = torch.stack(gram_det, dim=1)

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    return res
