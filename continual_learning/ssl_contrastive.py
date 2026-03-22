import torch
import torch.nn.functional as F

class ContrastiveSSL:
    """
    Mathematical Model of Self-Supervised Contrastive Loss (L_SSL)
    Enforces robustness to Concept Drift by dispersing feature representations
    via InfoNCE pseudo-augmentation contrast mapping.
    
    Formula Reference:
    L_SSL = -log( exp(sim(z_i, z_i')) / sum(exp(sim(z_i, z_j))) )
    """
    def __init__(self, model, temperature: float = 0.1, noise_std: float = 0.05):
        self.model = model
        self.temperature = temperature
        self.noise_std = noise_std
        
    def loss(self, X: torch.Tensor) -> torch.Tensor:
        """Computes the Contrastive Loss penalty for the current streaming batch"""
        
        # 1. Generate pseudo-augmentation (noisy perturbation)
        X_aug = X + torch.randn_like(X) * self.noise_std
        
        # 2. Extract latent feature representation space
        z1 = F.normalize(self.model.extract_features(X), dim=1)
        z2 = F.normalize(self.model.extract_features(X_aug), dim=1)
        
        batch_size = z1.size(0)
        
        # 3. Calculate positive pairs (cosine similarity / temp)
        pos_sim = (z1 * z2).sum(dim=-1) / self.temperature
        
        # 4. Calculate negative pairs against all augmented variants
        neg_sim = torch.matmul(z1, z2.T) / self.temperature
        
        # Mask identity bounds
        mask = torch.eye(batch_size, device=X.device, dtype=torch.bool)
        neg_sim = neg_sim.masked_fill(mask, -9e15)
        
        # 5. Compile InfoNCE Ratio
        numerator = torch.exp(pos_sim)
        denominator = numerator + torch.exp(neg_sim).sum(dim=1)
        
        l_ssl = -torch.log(numerator / denominator).mean()
        
        return l_ssl
