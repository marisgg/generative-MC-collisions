import torch
import torch.nn as nn
import torch.nn.functional as F

# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, leaky = False, hidden_size = 750, device = "cpu"):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(input_dim, hidden_size)
        self.e2 = nn.Linear(hidden_size, hidden_size)

        if leaky:
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu

        # self.activation = torch.tanh

        self.mean = nn.Linear(hidden_size, latent_dim)
        self.log_std = nn.Linear(hidden_size, latent_dim)

        self.d1 = nn.Linear(input_dim + latent_dim, hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)
        self.d3 = nn.Linear(hidden_size, input_dim)

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device


    def forward(self, input):
        z = self.activation(self.e1(input))
        z = self.activation(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        u = self.decode(input, z)

        return u, mean, std

    def loss_function(self, input):
        # Variational Auto-Encoder Training
        recon, mean, std = self.forward(input)
        recon_loss = F.mse_loss(recon, input)
        KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss
        return vae_loss, recon_loss, KL_loss


    def decode(self, input, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((self.input_dim, self.latent_dim)).to(self.device).clamp(-0.5,0.5)

        a = self.activation(self.d1(torch.cat([input, z], 1)))
        a = self.activation(self.d2(a))
        return self.d3(a)