import torch
def critic_loss_gp(x_fake, x_true, critic, idx, alpha=1, lambda_val=10):
  """The loss function for the critric.
  We don't need the gradient to accumulate to the G, because x~ = D(G(z)). So, we should detach to avoid gradient accumulation.
  We should detach the x_fake and x_true before passing into this (for explicit)
  """
  disc_fake = critic(x_fake, alpha=alpha, idx=idx).mean()
  disc_true = critic(x_true, alpha=alpha, idx=idx).mean()
  batch_size = x_fake.shape[0]
  epsilon = torch.rand(batch_size, 1, 1, 1, device=x_true.device) #Sample U(0,1)

  x_hat = epsilon * x_true + (1 - epsilon) * x_fake

  x_hat.requires_grad_(True)
  disc_xhat = critic(x_hat, alpha=alpha, idx=idx)
  # Gradient penalty
  grad = torch.autograd.grad(
      outputs=disc_xhat,
      inputs=x_hat,
      create_graph=True,
      retain_graph=True,
      grad_outputs=torch.ones_like(disc_xhat), #Because the critic have multiple outputs
      # And the grad_outputs is the vector in the jacobian vector trick.
  )[0] #Now the grad is a vector of gradient of Dw with respect to x_hat

  # The L2 norm of the full gradient vector of each sample.
  grad = grad.view(batch_size, -1) # flatten the grad
  grad_norm = grad.norm(2, dim=1)
  grad_term = lambda_val * ((grad_norm - 1)**2).mean() # Got a scalar value
  return disc_fake - disc_true + grad_term # Get just a scalar value

def gen_loss(x_fake, critic, idx, alpha=1):
  """
  Args:
  - x_fake: the G(z)
  """
  disc_fake = critic(x_fake, alpha=alpha, idx=idx).mean()
  return -disc_fake



