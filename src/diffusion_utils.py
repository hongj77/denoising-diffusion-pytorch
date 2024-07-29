import torch

def linear_beta_schedule(num_steps: int) -> torch.tensor:
    """The noise schedule determines how the variance changes at each step.

    There are many different types of noise schedules, but here we define a
    simple linear one from the paper.
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, num_steps)

def sample_x_t(x0: torch.tensor, noise: torch.tensor, t: torch.tensor, betas: torch.tensor) -> torch.tensor:
  """Return sample from a normal distribution for the forward process.

  Code implementation of x_t ~ p(x_t | x0).

   Args:
    x0: tensor of shape [b,c,h,w]
    noise: tensor of shape [b,c,h,w]
    t: tensor of shape shape [b,]. Timesteps are not necessarily the same for all batches.
    betas: tensor of shape [num_steps, ] with variance for each t.

   Returns:
    A tensor x_t with shape [b,c,h,w]
  """
  alphas = 1-betas
  alpha_bar = torch.cumprod(alphas, dim=0)

  # Get mean for each batch and reshape to [b,1,1,1] for broadcasting.
  mean = torch.sqrt(alpha_bar)
  mean = mean.gather(dim=-1, index=t.cpu())
  mean = mean.view(mean.shape[0], 1, 1, 1).to(t.device)


  # Get std for each batch and reshape to [b,1,1,1] for broadcasting.
  std = torch.sqrt(1-alpha_bar)
  std = std.gather(dim=-1, index=t.cpu())
  std = std.view(std.shape[0], 1, 1, 1).to(t.device)

  # Sampling from a Normal random variable.
  x_t_sample = mean * x0 + std * noise

  return x_t_sample

  #@title backward process
import einops

def gather_and_reshape(tensor, indices, out_shape):
  """
  Convenience function to index into tensor with indices then reshape to out_shape.
  """
  gathered = tensor.gather(dim=0, index=indices.cpu())
  gathered = gathered.view(out_shape).to(tensor.device)
  return gathered

def sample_body(model, x, step, betas, alpha_bar, label):
  """
   - step is shape [b, ]
  """
  batch_size = x.shape[0]
  out_shape = (batch_size, 1,1,1)

  # Index into each of these tensors with our `step` but we need the output result to have shape (b,1,1,1)
  alpha_bar_t = gather_and_reshape(alpha_bar, step, out_shape)
  betas_t = gather_and_reshape(betas, step, out_shape)
  alphas_t = 1-betas_t

  mean = (x - (betas_t / torch.sqrt(1-alpha_bar_t)) * model(x, step, label)) / torch.sqrt(alphas_t)

  if step[0] == 0:
    return mean

  variance = betas_t
  noise = torch.randn_like(x)
  return mean + torch.sqrt(variance)*noise

  # x_t_sample = mean * x0 + std * noise
  return x

@torch.no_grad()
def sample_images(model, num_steps, batch_size, img_size, num_channels, label):
  # Start with pure gaussian noise.
  x = torch.randn((batch_size, num_channels, img_size, img_size), device=model.device)
  betas = linear_beta_schedule(num_steps)
  alphas = 1-betas
  alpha_bar = torch.cumprod(alphas, dim=0)
  # Denoise loop for T steps.
  intermediate_steps = []
  for step in reversed(range(0, num_steps)):
    step_repeated = einops.repeat(torch.tensor(step), '-> b', b=batch_size).to(model.device)
    x = sample_body(model, x, step_repeated, betas, alpha_bar, label)
    intermediate_steps.append(x.cpu())
  return intermediate_steps