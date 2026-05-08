import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

def visualize_feature_maps(features, num_maps=16):
    """
    Visualize internal feature maps from the encoder.
    features: Tensor of shape (1, C, H, W)
    """
    if features.dim() == 4:
        features = features.squeeze(0)
    
    C, H, W = features.shape
    num_maps = min(C, num_maps)
    
    fig, axes = plt.subplots(int(np.ceil(num_maps/4)), 4, figsize=(12, 3 * np.ceil(num_maps/4)))
    fig.suptitle('Encoder Feature Maps Activation', fontsize=16)
    
    for i, ax in enumerate(axes.flatten()):
        if i < num_maps:
            f_map = features[i].detach().cpu().numpy()
            ax.imshow(f_map, cmap='viridis')
            ax.axis('off')
        else:
            ax.axis('off')
            
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def visualize_color_distribution(colour_dist):
    """
    Visualize the predicted color distribution in the 313-bin ab space.
    colour_dist: Tensor of shape (1, 313, H, W)
    """
    # Average distribution across spatial dimensions
    avg_dist = colour_dist.mean(dim=(2, 3)).squeeze(0).detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(313), avg_dist, color='royalblue', alpha=0.8)
    ax.set_title('Average Predicted Color Distribution (313 Bins)')
    ax.set_xlabel('Quantized AB Bin Index')
    ax.set_ylabel('Probability Logit')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def visualize_entropy_confidence(colour_dist):
    """
    Calculate and visualize entropy-based confidence map.
    colour_dist: Tensor of shape (1, 313, H, W)
    """
    probs = F.softmax(colour_dist, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    
    # Normalize entropy to 0-1 confidence (inverse of entropy)
    max_entropy = torch.log(torch.tensor(313.0)).item()
    confidence = 1.0 - (entropy / max_entropy)
    
    conf_map = confidence.squeeze(0).detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(conf_map, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_title('Entropy-based Confidence Map')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Confidence')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def visualize_attention_map(attention_weights, original_image):
    """
    Visualize self-attention maps over the image.
    attention_weights: Tensor of shape (1, num_heads, H, W) or aggregated.
    """
    attn_map = attention_weights.mean(dim=1).squeeze(0).detach().cpu().numpy()
    
    # Resize to match original image
    attn_map_resized = np.array(Image.fromarray(attn_map).resize(original_image.size, Image.BILINEAR))
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(original_image)
    ax.imshow(attn_map_resized, cmap='jet', alpha=0.5)
    ax.set_title('Self-Attention / Color Query Map')
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return Image.open(buf)
