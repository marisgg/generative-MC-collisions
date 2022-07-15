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

    def encode(self, input):
        z = self.activation(self.e1(input))
        z = self.activation(self.e2(z))
        return z


    def forward(self, input):
        z = self.encode(input)

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        u = self.decode(input, z)

        return u, mean, std

    def loss_function(self, input):
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

class Generator(nn.Module):
    def __init__(self, input_shape, latent_dim = 100):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, input_shape),
            nn.Tanh()
        )

    def forward(self, noise):
        return self.model(noise)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.model(input)

class GAN(nn.Module):
    def __init__(self, input_shape, latent_dim = 100):
        super(GAN, self).__init__()
        self.input_shape = input_shape
        self.latent_dim  = latent_dim

        self.discriminator = Discriminator(self.input_shape)
        self.generator     = Generator(self.input_shape, latent_dim = self.latent_dim)

        self.adversarial_loss = torch.nn.BCELoss()

        self.optimizer_G = torch.optim.Adam(self.generator.parameters())#, lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters())#, lr=opt.lr, betas=(opt.b1, opt.b2))

    def generate(self, size):
        z = torch.Tensor(torch.normal(0, 1, size=(size, self.latent_dim)))
        return self.generator(z)
    
    def forward(self, input):
        gen_data = self.generate(input.size(0))
        fake_cls = self.discriminator(gen_data.detach())
        real_cls = self.discriminator(input) 
        return gen_data, fake_cls, real_cls

    def train_with_batch(self, input):
        self.ones  = torch.Tensor(input.size(0), 1).fill_(1.0).requires_grad_(False)
        self.zeros = torch.Tensor(input.size(0), 1).fill_(0.0).requires_grad_(False)
        # INIT
        self.discriminator.train()
        self.generator.train()
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

        # FORWARD PASS
        gen_data, fake_cls, real_cls = self.forward(input)

        # TRAIN GENERATOR
        gen_data_2 = self.discriminator(gen_data)
        g_loss = self.adversarial_loss(gen_data_2, self.ones)

        g_loss.backward()
        self.optimizer_G.step()

        # TRAIN DISCRIMINATOR
        # Measure discriminator's ability to classify real from generated samples
        real_loss = self.adversarial_loss(real_cls, self.ones)
        fake_loss = self.adversarial_loss(fake_cls, self.zeros)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        self.optimizer_D.step()

        return d_loss.item(), g_loss.item()