import torch
import config

# Variable to read the constant values
conf = config.Config

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """
    Computes gradient penalty that helps stabilize the magnitude of the gradients that the
    discriminator provies to the generator, and thus help stabilize the training of the generator.
    """
    # Get random interpolations between real and fake samples
    alpha = torch.rand(real_samples.size(0), 1, 1, 1)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    interpolates = interpolates.requires_grad_(True)

    # Get the discriminator output for the interpolations
    d_interpolates = discriminator(interpolates)

    # Get gradients w.r.t the interpolations
    fake = torch.ones(real_samples.size(0), 1)
    gradients = torch.autograd.grad(
        outputs = d_interpolates,
        inputs = interpolates,
        grad_outputs = fake,
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    # Compute gradient penalty
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()

    return gradient_penalty


def train_one_step(generator, discriminator, d_optimizer, g_optimizer, real_samples):
    """ Trains the network for one step"""

    # Samples from the lantent distribution
    latent = torch.randn(conf.BATCH_SIZE, conf.LATENT_DIM)

    # TRAIN THE DISCRIMINATOR
    # Reset cached gradients to zero
    d_optimizer.zero_grad()

    # Get discriminator outputs for the real samples
    pred_real = discriminator(real_samples)
    # Compute the loss function
    d_loss_real = -torch.mean(pred_real)
    # Backpropogate the gradients
    d_loss_real.backward()

    # Generate fake samples with the generator
    fake_samples = generator(latent)
    # Get discriminator outputs for the fake samples
    pred_fake_d = discriminator(fake_samples.detach())
    # Compute the loss
    d_loss_fake = torch.mean(pred_fake_d)
    # Backpropogate the gradients
    d_loss_fake.backward()

    # Compute gradient penalty
    gradient_penalty = 10.0 * compute_gradient_penalty(discriminator, real_samples.data, fake_samples.data)
    # Backpropogate the gradients
    gradient_penalty.backward()

    # Update the weights
    d_optimizer.step()


    # TRAIN THE GENERATOR
    # Reset cached gradients to zero
    g_optimizer.zero_grad()
    # Get discriminator outputs for the fake samples
    pred_fake_g = discriminator(fake_samples)
    # Compute the loss
    g_loss = -torch.mean(pred_fake_g)
    # Backpropogate the gradients
    g_loss.backward()
    # Update the weights
    g_optimizer.step()

    return d_loss_fake + d_loss_real, g_loss



