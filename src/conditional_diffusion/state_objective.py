"""Matched bounded-state parameterizations, with a single semantic output head.

This is a training/inference-consistency screen, not a claim of a new generic
diffusion formulation. No historical source/config/checkpoint is changed.
"""
import torch
from .static_semantic import StaticSemanticModel


def bounded_state(logits, mapping):
    if mapping == 'tanh':
        return logits.tanh()
    if mapping == 'simplex':
        return 2*logits.softmax(1)-1
    raise ValueError('Unknown bounded state parameterization')


class StateObjectiveModel(StaticSemanticModel):
    def __init__(self, mapping, seed=42):
        if mapping not in ['tanh','simplex']:
            raise ValueError(mapping)
        super().__init__('S2', seed=seed)
        self.state_mapping = mapping

    def forward_with_rgb(self, features, state=None, t=None):
        logits = super().forward_with_rgb(features, state, t)
        # One learned head. Both denoising states are fixed, differentiable maps
        # of the same logits; there is no extra classifier or output capacity.
        return dict(logits=logits, x0=bounded_state(logits,self.state_mapping))


@torch.no_grad()
def sample_scores(model, diffusion, image, noise, steps=5):
    """Eta0 DDIM using bounded x0, returning raw final semantic logits in both arms."""
    if not isinstance(model,StateObjectiveModel):
        raise TypeError('This sampler requires the registered state-objective model')
    if model.timesteps != diffusion.timesteps or not 1 <= steps <= diffusion.timesteps:
        raise ValueError('Invalid timestep grid')
    if noise.shape != (len(image),4,*image.shape[-2:]) or noise.device != image.device:
        raise ValueError('Noise must match the RGB batch/spatial shape/device')
    if diffusion.abar.device != image.device:
        raise ValueError('Diffusion schedule device mismatch')
    features = model.encode_rgb(image)
    grid = torch.linspace(diffusion.timesteps-1,0,steps,device=image.device).round().long()
    xt = noise.clone()
    for index,t in enumerate(grid):
        output = model.forward_with_rgb(features,xt,t.expand(len(image)))
        x0 = output['x0'].clamp(-1,1)
        if index == len(grid)-1:
            return output['logits']
        a,ap = diffusion.abar[t],diffusion.abar[grid[index+1]]
        epsilon = (xt-a.sqrt()*x0)/(1-a).sqrt()
        xt = ap.sqrt()*x0+(1-ap).sqrt()*epsilon
    raise RuntimeError('Empty sampler schedule')


def build_state_model(config):
    if config['mode'] != 'S2' or config['state_mapping'] not in ['tanh','simplex']:
        raise ValueError('Unregistered state-objective configuration')
    return StateObjectiveModel(config['state_mapping'],config['seed']).to(config['device'])


def bind_experiment():
    """Explicit process-local dependency injection into the frozen shared loop.

    The existing trainer's losses/recycling already accept {logits,x0}. Reuse its
    audited logging, budgets, atomic checkpoints, resume and native evaluation;
    replace only its model factory and cached sampler in this dedicated worker.
    This function is never called by historical launchers or on module import.
    """
    from . import static_experiment
    static_experiment.build = build_state_model
    static_experiment.sample_cached = sample_scores
