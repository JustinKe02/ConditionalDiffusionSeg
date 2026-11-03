import math
import torch
from torch.nn import functional as F


class GaussianDiffusion:
    def __init__(self, timesteps=1000, device='cpu'):
        self.timesteps = timesteps
        x = torch.linspace(0, timesteps, timesteps+1, dtype=torch.float64)
        abar = torch.cos(((x/timesteps+.008)/1.008)*math.pi/2).square()
        abar = abar/abar[0]
        betas = (1-abar[1:]/abar[:-1]).clamp(0, .999)
        self.abar = (1-betas).cumprod(0).float().to(device)

    def training_timesteps(self, batch_size, policy='uniform', high_noise_start=900):
        """Uniform baseline or an explicit 50/50 high-noise P0 ablation.

        Uniform retains the original RNG call. Stratified sampling deliberately
        changes the training-time distribution; it is not an unbiased estimator
        of the baseline objective. RNG state alone suffices for exact resume.
        """
        if policy == 'uniform':
            return torch.randint(self.timesteps, (batch_size,), device=self.abar.device)
        if policy != 'high_noise_stratified':
            raise ValueError('Unknown timestep sampling policy')
        if batch_size < 2 or batch_size % 2 or not 0 < high_noise_start < self.timesteps:
            raise ValueError('Stratification requires an even batch >=2 and a valid split')
        half = batch_size//2
        low = torch.randint(high_noise_start, (half,), device=self.abar.device)
        high = torch.randint(high_noise_start, self.timesteps, (half,), device=self.abar.device)
        return torch.cat([low, high])

    @staticmethod
    def encode(mask):
        return 2*F.one_hot(mask, 4).permute(0, 3, 1, 2).float()-1

    def q_sample(self, x0, t, noise):
        a = self.abar[t][:, None, None, None]
        return a.sqrt()*x0+(1-a).sqrt()*noise

    @torch.no_grad()
    def sample(self, model, image, noise, steps=20):
        if not 1 <= steps <= self.timesteps:
            raise ValueError('Sampling steps must lie between 1 and T')
        grid = torch.linspace(self.timesteps-1, 0, steps, device=image.device).round().long()
        xt = noise.clone()
        for i, t in enumerate(grid):
            batch_t = t.expand(image.shape[0])
            output = model(image, xt, batch_t)
            x0 = (output['x0'] if isinstance(output, dict) else output).clamp(-1, 1)
            if i == len(grid)-1:
                # Semantic scores only at the final step; legacy returns clipped x0.
                return output['logits'] if isinstance(output, dict) else x0
            a, ap = self.abar[t], self.abar[grid[i+1]]
            epsilon = (xt-a.sqrt()*x0)/(1-a).sqrt()
            xt = ap.sqrt()*x0+(1-ap).sqrt()*epsilon
        raise RuntimeError('Empty sample schedule')
