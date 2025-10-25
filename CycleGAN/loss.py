import torch
import torch.nn as nn

# The GAN loss (use WGAN-GP)
# The Discriminator loss

def disc_loss(true_out, fake_out, disc):
  """Returns the discriminator loss.
  Args:
  - true_out: either imgA or imgB
  - fake_out: G(imgA) or F(imgB)
  - disc: discriminator either D_g or D_f
  imgA -> G(imgA) -> D_g(G(imgA))
  imgB -> F(imgB) -> D_f(F(imgB))
  Need to pass the right order.
  If true_out is imgA, then fake_out is G(imgA) and disc is D_g
  if true_out is imgB, then fake_out is F(imgB) and disc is D_F
  """
  true_loss = ((disc(true_out) - 1)**2).mean()
  fake_loss = ((disc(fake_out))**2).mean()

  return (true_loss + fake_loss)/2

def total_disc_loss(Dg, Df, imgA, imgB, fakeA, fakeB):
    """
        Dg, Df: discriminators
        imgA, imgB: images from domain A, domain B.
        fakeA, fakeB: generated A and B or F(imgB), G(imgA) - pass it directly instead of passing the generator F, G and then calcuate it everytime in this function.
    """

    #   disc_loss_g = disc_loss(imgA, fakeA, Dg)
    #   disc_loss_f = disc_loss(imgB, fakeB, Df)
    loss_A = disc_loss(imgA, fakeA, Df)  # judge real A vs fake A: use Df
    loss_B = disc_loss(imgB, fakeB, Dg)  # judge real B vs fake B: use Dg
    return loss_A + loss_B
def gen_gan_loss(G, F, Dg, Df, imgA, imgB):
  """Returns the generator loss of dimension
  Args:
  - D, G: generators
  - img: either from X or Y domain
  - Need to the other correctly, if img from X, then pass (D, G, img)
  - If img from Y: then pass (G, D, img)
  """
  # Lgan(G,D,X,Y)
  genG = ((Dg(G(imgA)) - 1)**2).mean() #fakeB = G(imgA): judge real B vs fake B: use Dg
  # Lgan(D,G,Y,X)
  genF = ((Df(F(imgB)) - 1)**2).mean() #fakeA = F(imgB): judge real A vs fake A: use Df

  return genG + genF

l1_loss = nn.L1Loss()
def cycle_loss(G, F, Dg, Df, imgA, imgB):
    """
    Cycle-consistency loss for CycleGAN.

    Args:
        G: generator from A -> B
        F: generator from B -> A
        imgA: image from domain A
        imgB: image from domain B
    """
    # Forward cycle: A -> G -> F -> A
    cycleA = F(G(imgA))
    lossA = l1_loss(cycleA, imgA)

    # Backward cycle: B -> F -> G -> B
    cycleB = G(F(imgB))
    lossB = l1_loss(cycleB, imgB)

    return lossA + lossB

# The identity loss
def identity_loss(G, F, imgA, imgB):
    """
    Identity loss for CycleGAN.

    Args:
        gen: Generator (G or F)
        img: Input image (from either domain A or B)
    imgA -> G -> fakeB -> F -> fakeA
    Returns:
        L1 identity loss
    """
    identityA = F(imgA)
    lossA = l1_loss(identityA, imgA)

    identityB = G(imgB)
    lossB = l1_loss(identityB, imgB)

    return lossA + lossB

def total_gen_loss(G, F, Dg, Df, imgA, imgB, lambda_cyc=10.0, lambda_idt=5.0):
  """
  Cycle 1: imgA -> G(imgA) -> F(G(imgA)) ≈ imgA
  Cycle 2: imgB -> F(imgB) -> G(F(imgB)) ≈ imgB
  Args:
  G, F: generators
  D: discriminator
  imgA: image from domain A - horses
  imgB: image from domain B - zebras
  """
  gan_loss_val = gen_gan_loss(G, F, Dg, Df, imgA, imgB)
  cycle_loss_val = cycle_loss(G, F, Dg, Df, imgA, imgB)
  identity_loss_val = identity_loss(G, F, imgA, imgB)
  return gan_loss_val + lambda_cyc * cycle_loss_val + lambda_idt * identity_loss_val