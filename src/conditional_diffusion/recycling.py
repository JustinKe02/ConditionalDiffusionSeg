"""Optional predicted-state training; ground truth remains the loss target."""
import torch


def mixed_recycling_input(model, diffusion, image, x0, t, noise, fraction=0.5):
    """Recycle exactly half a batch, with membership independent of time strata.

    The teacher is the same current model, evaluated once at terminal time from
    independent pure Gaussian noise. Its bounded prediction is detached and then
    noised using the regular training noise. No teacher loss or gradient is used.
    """
    batch = x0.shape[0]
    if fraction != 0.5 or batch < 2 or batch % 2:
        raise ValueError('Registered recycling requires an even batch and fraction=0.5')
    selected = torch.randperm(batch, device=x0.device)[:batch // 2]
    with torch.no_grad():
        teacher_noise = torch.randn_like(x0[selected])
        terminal = torch.full_like(t[selected], diffusion.timesteps - 1)
        output = model(image[selected], teacher_noise, terminal)
        predicted = (output['x0'] if isinstance(output, dict) else output).clamp(-1, 1)
    source = x0.clone()
    source[selected] = predicted.detach()
    noisy = diffusion.q_sample(source, t, noise)
    return noisy, selected
