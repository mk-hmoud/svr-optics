import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Generator(nn.Module):
    def __init__(self, input_dim=8, output_dim=8):
        super(Generator, self).__init__()
        # 5 hidden layers, 128 neurons each (Table I)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, z):
        return self.model(z)

class Critic(nn.Module):
    def __init__(self, input_dim=8):
        super(Critic, self).__init__()
        # 5 hidden layers, 32 neurons each (Table I specifies BatchNorm for Critic)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.model(x)

def compute_gradient_penalty(critic, real_samples, fake_samples):
    alpha = torch.rand((real_samples.size(0), 1))
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)
    fake = torch.ones((real_samples.size(0), 1), requires_grad=False)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_wgan_paper(train_data, epochs=2500, batch_size=32, n_critic=5, lr=0.0001):
    data_tensor = torch.FloatTensor(train_data)
    input_dim = 8
    output_dim = 8
    
    generator = Generator(input_dim, output_dim)
    critic = Critic(output_dim)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_C = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))
    
    for epoch in range(epochs):
        for i in range(0, len(data_tensor), batch_size):
            if i + batch_size > len(data_tensor): break
                
            for _ in range(n_critic):
                optimizer_C.zero_grad()
                idx = np.random.randint(0, data_tensor.size(0), batch_size)
                real_samples = data_tensor[idx]
                z = torch.randn(batch_size, input_dim)
                fake_samples = generator(z).detach()
                loss_C = -torch.mean(critic(real_samples)) + torch.mean(critic(fake_samples))
                gp = compute_gradient_penalty(critic, real_samples, fake_samples)
                loss_C += 10 * gp
                loss_C.backward()
                optimizer_C.step()
            
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, input_dim)
            gen_samples = generator(z)
            loss_G = -torch.mean(critic(gen_samples))
            loss_G.backward()
            optimizer_G.step()
            
        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss C: {loss_C.item():.4f} | Loss G: {loss_G.item():.4f}")
            
    return generator

def generate_samples_paper(generator, num_samples=1000):
    generator.eval()
    z = torch.randn(num_samples, 8)
    with torch.no_grad():
        samples = generator(z).numpy()
    return samples
