import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import Adam
import torchvision.models as models
from scipy import linalg


# Configuration
class Config:
    def __init__(self):
        # Data parameters
        self.data_path = "extracted_data/Samples"
        self.image_size = 128  
        self.channels = 1  # Grayscale images

        # Training parameters
        self.batch_size = 32
        self.epochs = 100
        self.lr = 2e-4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # DDPM parameters
        self.timesteps = 1000
        self.beta_start = 1e-4
        self.beta_end = 0.02

        # Model parameters
        self.hidden_dims = [64, 128, 256, 512]
        self.num_res_blocks = 2

config = Config()

# Dataset class
class LensingDataset(Dataset):
    # Pass image_size to the constructor
    def __init__(self, data_path, image_size, transform=None):
        self.data_path = data_path
        self.file_list = [f for f in os.listdir(data_path) if f.endswith('.npy')]
        self.image_size = image_size # Store image size

        # Define a default transform sequence including resize and normalization
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(), # Converts numpy HWC/HW [0,1] to CHW [0,1] Tensor
                # Ensure resizing happens after ToTensor if input is numpy
                transforms.Resize((self.image_size, self.image_size), antialias=True),
                # Normalize to [-1, 1] which is common for diffusion models
                transforms.Normalize((0.5,), (0.5,)) # (mean,), (std,) for grayscale
            ])
        else:
            # If a custom transform is provided, ensure it handles resizing and normalization
            self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_list[idx])
        # Load as float32
        image = np.load(file_path).astype(np.float32)
        
        # Normalize to [0, 1] first, required by transforms.ToTensor
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        else:
            # Handle constant images (e.g., all zeros)
            image = np.zeros_like(image)

        # Handle different shapes
        if len(image.shape) == 1:
            # If image is 1D, try to reshape to a square
            size = int(np.sqrt(image.shape[0]))
            if size * size == image.shape[0]:
                image = image.reshape(size, size)
            else:
                # If not a perfect square, resize to target dimensions
                # First reshape to any 2D shape that fits
                closest_factor = int(np.sqrt(image.shape[0]))
                while image.shape[0] % closest_factor != 0 and closest_factor > 1:
                    closest_factor -= 1
                if closest_factor > 1:
                    other_dim = image.shape[0] // closest_factor
                    image = image.reshape(closest_factor, other_dim)
                else:
                    # If no clean factors, just use a simple reshape and accept distortion
                    image = np.resize(image, (self.image_size, self.image_size))
        elif len(image.shape) == 3:
            # If 3D, take the first slice or average across slices
            if image.shape[0] == 150:
                # Take the first slice as a representative image
                image = image[0]
            else:
                # Average across the first dimension
                image = np.mean(image, axis=0)
        
        # Ensure image is 2D before transform
        if len(image.shape) != 2:
            # Last resort: resize to target dimensions
            image = np.resize(image, (self.image_size, self.image_size))
            
        if self.transform:
            # Apply transforms (ToTensor, Resize, Normalize)
            image = self.transform(image)

        # Ensure final tensor is float32 (ToTensor usually handles this)
        return image.float()



#  The diffusion schedule
def get_noise_schedule(config):
    """Linear beta schedule for diffusion model."""
    beta = np.linspace(config.beta_start, config.beta_end, config.timesteps)
    sqrt_beta = np.sqrt(beta)
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)
    sqrt_alpha_bar = np.sqrt(alpha_bar)
    sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bar)

    # Convert to torch tensors (explicitly as float)
    beta = torch.tensor(beta, dtype=torch.float32).to(config.device)
    sqrt_beta = torch.tensor(sqrt_beta, dtype=torch.float32).to(config.device)
    alpha = torch.tensor(alpha, dtype=torch.float32).to(config.device)
    alpha_bar = torch.tensor(alpha_bar, dtype=torch.float32).to(config.device)
    sqrt_alpha_bar = torch.tensor(sqrt_alpha_bar, dtype=torch.float32).to(config.device)
    sqrt_one_minus_alpha_bar = torch.tensor(sqrt_one_minus_alpha_bar, dtype=torch.float32).to(config.device)

    return {
        'beta': beta,
        'sqrt_beta': sqrt_beta,
        'alpha': alpha,
        'alpha_bar': alpha_bar,
        'sqrt_alpha_bar': sqrt_alpha_bar,
        'sqrt_one_minus_alpha_bar': sqrt_one_minus_alpha_bar,
    }

# Sinusoidal position embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :] # Ensure time is float for multiplication
        embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)
        # If dim is odd, pad the final dimension
        if self.dim % 2 == 1:
             embeddings = F.pad(embeddings, (0,1)) # Pad last dimension by 1
        return embeddings

# Attention module
class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4): # Added num_heads argument
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        # Ensure embed_dim is divisible by num_heads
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True) # Use num_heads
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = x.view(-1, self.channels, size[0] * size[1]).swapaxes(1, 2) # B, H*W, C
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size[0], size[1]) # B, C, H, W

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, groups=8): # Added groups parameter
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_mlp = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        # Ensure out_channels >= groups for GroupNorm
        self.norm1 = nn.GroupNorm(min(groups, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(groups, out_channels), out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU() # Use SiLU consistently

    def forward(self, x, time_emb=None):
        # print(f"ResBlock input shape: {x.shape}, in_channels: {self.in_channels}, out_channels: {self.out_channels}")  (for debugging ... )
        
        h = self.act(self.norm1(self.conv1(x)))

        if time_emb is not None and self.time_mlp is not None:
            # time_emb is expected to be (B, time_emb_dim)
            time_emb_proj = self.act(self.time_mlp(time_emb)) # Project time embedding
            # Add broadcasted time embedding: (B, C, 1, 1)
            h = h + time_emb_proj.view(-1, time_emb_proj.shape[1], 1, 1)

        h = self.act(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)

# U-Net architecture for noise prediction
class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.channels = config.channels
        self.time_dim = config.hidden_dims[0] # Time embedding dimension often matches first hidden dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim * 4), # Project to larger dimension
            nn.SiLU(),
            nn.Linear(self.time_dim * 4, self.time_dim * 4) # Keep projected dimension for ResBlocks
        )
        time_emb_dim_unet = self.time_dim * 4 # Dimension passed to ResBlocks

        # Initial projection
        init_dim = config.hidden_dims[0]
        self.init_conv = nn.Conv2d(config.channels, init_dim, 3, padding=1)

        # Encoder
        self.downs = nn.ModuleList()
        dims = config.hidden_dims
        current_dim = init_dim
        
        for i in range(len(dims)):
            # Use ModuleList for layers within each block for clarity
            down_block_layers = nn.ModuleList()

            # Add ResNet blocks
            for _ in range(config.num_res_blocks):
                down_block_layers.append(ResidualBlock(current_dim, current_dim, time_emb_dim=time_emb_dim_unet))

            # Add Self-Attention at lower resolutions (e.g., 16x16, 8x8 if applicable)
            if i >= (len(dims) - 2): # Add attention for last two blocks
                down_block_layers.append(SelfAttention(current_dim))

            # Downsample (if not the last block)
            if i < len(dims) - 1:
                down_block_layers.append(nn.Conv2d(current_dim, dims[i+1], kernel_size=4, stride=2, padding=1))
                current_dim = dims[i+1]

            self.downs.append(down_block_layers)

        # Middle block
        mid_dim = dims[-1]
        self.mid = nn.ModuleList([
            ResidualBlock(mid_dim, mid_dim, time_emb_dim=time_emb_dim_unet),
            SelfAttention(mid_dim),
            ResidualBlock(mid_dim, mid_dim, time_emb_dim=time_emb_dim_unet)
        ])

        # Decoder
        self.ups = nn.ModuleList()
        for i in reversed(range(len(dims))):
            up_block_layers = nn.ModuleList()
            
            # Calculate correct input dimension for first ResBlock in each decoder block
            if i == len(dims) - 1:
                # First decoder block gets input from middle block (no skip yet)
                in_dim = dims[i]
            else:
                # Other blocks get input from previous block + skip connection
                in_dim = dims[i] * 2  # Previous block output + skip connection
            
            # First ResBlock after concatenation
            up_block_layers.append(ResidualBlock(in_dim, dims[i], time_emb_dim=time_emb_dim_unet))
            
            # Additional ResBlocks
            for _ in range(config.num_res_blocks - 1):
                up_block_layers.append(ResidualBlock(dims[i], dims[i], time_emb_dim=time_emb_dim_unet))
            
            # Add Self-Attention mirroring the encoder
            if i >= (len(dims) - 2):
                up_block_layers.append(SelfAttention(dims[i]))
            
            # Upsample (if not the first block of decoder)
            if i > 0:
                up_block_layers.append(nn.ConvTranspose2d(dims[i], dims[i-1], kernel_size=4, stride=2, padding=1))
            
            self.ups.append(up_block_layers)

        # Final layers
        final_dim = dims[0]
        self.final_conv = nn.Sequential(
            nn.GroupNorm(min(8, final_dim), final_dim),
            nn.SiLU(),
            nn.Conv2d(final_dim, config.channels, 1)
        )

    def forward(self, x, t):
        # Time embedding
        t = self.time_mlp(t)
        
        # Initial convolution
        x = self.init_conv(x)
        skip_connections = []
        
        # Encoder
        for down_block in self.downs:
            for layer in down_block:
                if isinstance(layer, ResidualBlock) or isinstance(layer, SelfAttention):
                    x = layer(x, t) if isinstance(layer, ResidualBlock) else layer(x)
                else:  # Conv2d for downsampling
                    # Store the skip connection before downsampling
                    skip_connections.append(x)
                    x = layer(x)
        
        # Middle
        for layer in self.mid:
            x = layer(x, t) if isinstance(layer, ResidualBlock) else layer(x)
        
        # Decoder with shape checking
        for i, up_block in enumerate(self.ups):
            # Skip connection handling is done inside the loop for each block
            if i > 0:  # No skip connection for the first decoder block
                skip = skip_connections.pop()
                
                # Ensure spatial dimensions match before concatenating
                if x.shape[2:] != skip.shape[2:]:
                    # Resize skip connection to match x's spatial dimensions
                    skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
                
                # Concatenate along channel dimension
                x = torch.cat([x, skip], dim=1)
            
            # Process through all layers in the block
            for layer in up_block:
                if isinstance(layer, ResidualBlock) or isinstance(layer, SelfAttention):
                    x = layer(x, t) if isinstance(layer, ResidualBlock) else layer(x)
                else:  # ConvTranspose2d for upsampling
                    x = layer(x)
        
        # Final convolution
        return self.final_conv(x)


# Diffusion model
class DiffusionModel:
    def __init__(self, config):
        self.config = config
        self.model = UNet(config).to(config.device)
        self.noise_schedule = get_noise_schedule(config)
        self.optimizer = Adam(self.model.parameters(), lr=config.lr)
        self.loss_fn = nn.MSELoss()

    def forward_diffusion(self, x_0, t):
        """Add noise to the input image according to the timestep t."""
        # Ensure input is float32
        x_0 = x_0.float().to(self.config.device) # Ensure on correct device
        noise = torch.randn_like(x_0, dtype=torch.float32, device=self.config.device)
        # Ensure schedule tensors are used correctly (already on device)
        sqrt_alpha_bar = self.noise_schedule['sqrt_alpha_bar'][t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.noise_schedule['sqrt_one_minus_alpha_bar'][t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise # Return noise for loss calculation

    def train_step(self, x_0):
        """Single training step."""
        self.optimizer.zero_grad()

        # Ensure input is float32 and on device
        x_0 = x_0.float().to(self.config.device)

        # Sample timesteps uniformly
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.config.timesteps, (batch_size,), device=self.config.device).long()

        # Forward diffusion process
        x_t, noise = self.forward_diffusion(x_0, t)

        # Predict noise
        predicted_noise = self.model(x_t, t) # Pass x_t and t

        # Loss
        loss = self.loss_fn(predicted_noise, noise) # Compare predicted noise to actual noise

        # Backward and optimize
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad() # Decorator for inference mode
    def sample(self, n_samples, size=None):
        """Sample new images using the reverse diffusion process."""
        self.model.eval() # Set model to evaluation mode

        if size is None:
            size = (self.config.channels, self.config.image_size, self.config.image_size)

        # Start from random noise ~ N(0, I)
        x = torch.randn((n_samples, *size), device=self.config.device, dtype=torch.float32)

        # Reverse diffusion process loop
        for t in tqdm(reversed(range(self.config.timesteps)), desc="Sampling", total=self.config.timesteps):
            t_tensor = torch.full((n_samples,), t, device=self.config.device, dtype=torch.long)

            # Predict noise using the model
            predicted_noise = self.model(x, t_tensor)

            # Get diffusion parameters for timestep t
            alpha_t = self.noise_schedule['alpha'][t]
            alpha_t_bar = self.noise_schedule['alpha_bar'][t]
            beta_t = self.noise_schedule['beta'][t]
            sqrt_one_minus_alpha_t_bar = self.noise_schedule['sqrt_one_minus_alpha_bar'][t]
            sqrt_recip_alpha_t = torch.sqrt(1.0 / alpha_t) # Precompute for efficiency

            # Calculate mean of p(x_{t-1} | x_t)
            mean = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_t_bar) * predicted_noise)

            if t > 0:
                # Calculate variance (sigma_t^2 * I)
                # Use sqrt_beta for standard deviation
                std_dev = self.noise_schedule['sqrt_beta'][t]
                noise = torch.randn_like(x, dtype=torch.float32)
                x = mean + std_dev * noise # Update x_{t-1}
            else:
                # For t=0, the variance is zero
                x = mean

        self.model.train() # Set model back to training mode
        # Clip or normalize samples to expected range [-1, 1] if needed, though normalization in dataset helps
        # x = torch.clamp(x, -1.0, 1.0)
        return x


# FID Implementation
class FID:
    def __init__(self, config):
        self.device = config.device
        # Use a pre-trained model for feature extraction
        # models.inception_v3 requires weights argument
        try:
            # Newer torchvision versions
            self.inception_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
        except TypeError:
            # Older torchvision versions
             self.inception_model = models.inception_v3(pretrained=True, transform_input=False)

        self.inception_model.fc = nn.Identity()  # Remove final FC layer (output features)
        self.inception_model.AuxLogits.fc = nn.Identity() # Remove aux classifier head too
        self.inception_model = self.inception_model.to(self.device)
        self.inception_model.eval()


    @torch.no_grad() # Ensure no gradients are computed
    def extract_features(self, loader):
        """Extract features from the InceptionV3 model."""
        features = []

        for imgs in tqdm(loader, desc="Extracting features"):
            # Ensure float32 and on correct device
            imgs = imgs.float().to(self.device)

            # Preprocess for InceptionV3:
            # 1. Grayscale to RGB
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1, 3, 1, 1) # Repeat channel dim 3 times

            # 2. Resize to InceptionV3 input size (299x299)
            # Use antialias=True for better quality resizing
            imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False, antialias=True)

            # 3. Normalize according to InceptionV3's expected range [0, 1] and std normalization
            # The original code normalized to [-1, 1]. InceptionV3 usually expects [0, 1]
            # followed by its specific normalization. Let's assume input `imgs` are [-1, 1]
            # from the diffusion model/dataset. Rescale to [0, 1] first.
            imgs = (imgs + 1) / 2.0

            # Apply InceptionV3 specific normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            inception_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            imgs = inception_normalize(imgs)

            # Extract features (output is B x 2048 for InceptionV3 pool layer)
            # InceptionV3 returns InceptionOutputs object, get the main output
            feat = self.inception_model(imgs)
            if isinstance(feat, models.inception.InceptionOutputs):
                 feat = feat.logits # Main output features before FC layer

            features.append(feat.cpu().numpy()) # Move features to CPU and convert to numpy

        return np.concatenate(features, axis=0)

    def calculate_fid(self, real_features, generated_features):
        """Calculate Fréchet Inception Distance."""
        # Calculate mean and covariance statistics
        mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)

        # Calculate squared difference between means ||mu1 - mu2||^2
        sum_sq_diff = np.sum((mu1 - mu2)**2)

        # Calculate sqrt of product of covariances: sqrtm(sigma1 * sigma2)
        # Add small identity matrix for numerical stability if needed
        eps = 1e-6
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2) + eps * np.eye(sigma1.shape[0]), disp=False)

        # Handle potential complex numbers in sqrtm result
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        # Calculate FID score: ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2 * sqrtm(sigma1*sigma2))
        fid = sum_sq_diff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

        return fid


# Load dataset function (with error handling)
def load_dataset(config):
    try:
        # Pass image_size from config to the dataset constructor
        dataset = LensingDataset(config.data_path, config.image_size)
        print(f"Successfully loaded dataset with {len(dataset)} samples")
        # Check the first sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample shape: {sample.shape}, dtype: {sample.dtype}") # Should now be [1, 128, 128]
        else:
            print("Dataset is empty!")
            return None
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Check if data directory exists
        if not os.path.exists(config.data_path):
            print(f"Data directory '{config.data_path}' not found.")
            print("Please ensure the data is extracted correctly.")
            print("Expected structure: Samples/1.npy, Samples/2.npy, etc.")
        # List available files
        elif os.path.isdir(config.data_path):
            try:
                files = os.listdir(config.data_path)
                print(f"Files in {config.data_path}: {files[:10]}...")
                if not any(f.endswith('.npy') for f in files):
                    print("No .npy files found in the directory.")
            except Exception as list_e:
                print(f"Could not list files in {config.data_path}: {list_e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None

# Training function
def train(config):
    # Data loading
    dataset = load_dataset(config)
    if dataset is None:
        print("Dataset loading failed. Exiting training.")
        return None

    # Use more workers if CPU allows for faster data loading
    num_workers = 2
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if config.device=="cuda" else False)

    # Initialize model
    diffusion = DiffusionModel(config)

    # Training loop
    losses = []
    for epoch in range(config.epochs):
        epoch_losses = []
        # Use leave=True for the outer loop progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=True)

        for batch in progress_bar:
            # Batch is already on device if pin_memory=True and device=cuda
            # Otherwise move it here
            # batch = batch.to(config.device) # No need, handled in train_step
            loss = diffusion.train_step(batch)
            epoch_losses.append(loss)

            # Show running average loss in progress bar
            progress_bar.set_postfix({"loss": np.mean(epoch_losses[-20:])}) # Use last 20 batches for smoother avg

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{config.epochs}, Average Loss: {avg_loss:.4f}")

        # Generate and save samples periodically (e.g., every 10 epochs)
        if (epoch + 1) % 10 == 0 or epoch == config.epochs - 1:
            print(f"Generating samples for epoch {epoch+1}...")
            samples = diffusion.sample(16) # Generate 16 samples
            samples = (samples + 1) / 2  # Rescale from [-1, 1] to [0, 1] for visualization
            samples = samples.clamp(0.0, 1.0) # Clamp to ensure valid range

            plt.figure(figsize=(10, 10))
            for i in range(16):
                plt.subplot(4, 4, i+1)
                # Use viridis colormap, ensure image is HxW
                plt.imshow(samples[i, 0].cpu().numpy(), cmap='viridis')
                plt.axis('off')
            plt.suptitle(f"Samples Epoch {epoch+1}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
            save_path = f"samples_epoch_{epoch+1}.png"
            plt.savefig(save_path)
            print(f"Saved samples to {save_path}")
            plt.close() # Close the figure to free memory

        # Calculate FID score periodically (e.g., every 20 epochs)
        # FID calculation is expensive, don't do it too often
        if (epoch + 1) % 20 == 0 or epoch == config.epochs - 1:
            print(f"Calculating FID for epoch {epoch+1}...")
            fid_calculator = FID(config)

            # Extract features from real data (use a subset for efficiency if dataset is large)
            # Use a fixed subset for comparable FID scores across epochs
            subset_indices = np.random.choice(len(dataset), min(2048, len(dataset)), replace=False)
            real_subset = torch.utils.data.Subset(dataset, subset_indices)
            # Lower batch size for FID extraction to avoid OOM on GPU for Inception
            fid_batch_size = 16
            real_loader = DataLoader(real_subset, batch_size=fid_batch_size, shuffle=False, num_workers=num_workers)
            real_features = fid_calculator.extract_features(real_loader)

            # Generate samples and extract features
            n_fid_samples = len(real_subset) # Generate same number of samples as real subset
            generated_samples = []
            samples_generated = 0
            while samples_generated < n_fid_samples:
                 num_to_generate = min(fid_batch_size, n_fid_samples - samples_generated)
                 if num_to_generate <= 0: break
                 current_samples = diffusion.sample(num_to_generate)
                 generated_samples.append(current_samples.cpu()) # Move to CPU immediately
                 samples_generated += num_to_generate

            generated_samples_tensor = torch.cat(generated_samples, dim=0)
            # Create a simple TensorDataset for the DataLoader
            generated_dataset = torch.utils.data.TensorDataset(generated_samples_tensor)
            generated_loader = DataLoader(generated_dataset, batch_size=fid_batch_size, shuffle=False)
            generated_features = fid_calculator.extract_features(generated_loader)

            # Calculate FID
            fid_score = fid_calculator.calculate_fid(real_features, generated_features)
            print(f"Epoch {epoch+1}, FID Score: {fid_score:.2f}")

    # Save the trained model state dictionary
    model_save_path = "gravitational_lensing_diffusion.pth"
    torch.save(diffusion.model.state_dict(), model_save_path)
    print(f"Saved trained model to {model_save_path}")

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    loss_plot_path = "training_loss.png"
    plt.savefig(loss_plot_path)
    print(f"Saved training loss plot to {loss_plot_path}")
    plt.close()

    return diffusion

# Main function
def main():
    print(f"Using device: {config.device}")
    print("Training diffusion model for gravitational lensing...")
    diffusion_model = train(config) # Renamed variable to avoid conflict

    if diffusion_model is not None:
        # Generate final samples after training completion
        print("Generating final samples...")
        final_samples = diffusion_model.sample(16) # Generate 16 final samples
        final_samples = (final_samples + 1) / 2  # Rescale to [0, 1]
        final_samples = final_samples.clamp(0.0, 1.0)

        plt.figure(figsize=(12, 12))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.imshow(final_samples[i, 0].cpu().numpy(), cmap='viridis')
            plt.axis('off')
        plt.suptitle("Final Generated Samples", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        final_samples_path = "final_samples.png"
        plt.savefig(final_samples_path)
        print(f"Saved final generated samples to {final_samples_path}")
        # plt.show() # Optionally display the plot interactively

if __name__ == "__main__":
    main()