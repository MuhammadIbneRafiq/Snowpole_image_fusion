"""
Multi-source fusion quality metrics implementation.
These metrics evaluate fusion quality using all source images jointly rather than pairwise.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

def mef_ssim(fused, sources, sigma=1.5, C1=1e-4, C2=9e-4):
    """
    Multi-exposure fusion SSIM.
    
    Args:
        fused: HxW fused image
        sources: list of HxW source images
        
    Returns:
        Scalar SSIM score
    """
    # local means
    mu_f = gaussian_filter(fused, sigma)
    mu_s = [gaussian_filter(s, sigma) for s in sources]

    # local variances
    var_f = gaussian_filter(fused**2, sigma) - mu_f**2
    var_s = [gaussian_filter(s**2, sigma) - mu**2 for s, mu in zip(sources, mu_s)]

    # aggregate source statistics (multi-reference)
    mu_ref = np.mean(mu_s, axis=0)
    var_ref = np.mean(var_s, axis=0)

    # covariance between fused and reference
    cov_fr = gaussian_filter(fused * mu_ref, sigma) - mu_f * mu_ref

    # SSIM-like formula
    numerator = (2 * mu_f * mu_ref + C1) * (2 * cov_fr + C2)
    denominator = (mu_f**2 + mu_ref**2 + C1) * (var_f + var_ref + C2)

    ssim_map = numerator / (denominator + 1e-8)
    return np.mean(ssim_map)


def gradient_map(img):
    """Compute gradient magnitude map"""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)


def lgir_gradient_score(fused, sources, eps=1e-6):
    """
    LGIR-style gradient similarity score.
    
    Args:
        fused: HxW fused image
        sources: list of HxW source images
        
    Returns:
        Scalar gradient similarity score
    """
    gF = gradient_map(fused)
    gS = [gradient_map(s) for s in sources]

    # Local Intermediate Reference (dominant structure)
    g_ref = np.max(np.stack(gS, axis=0), axis=0)

    # similarity map
    sim_map = (2 * gF * g_ref + eps) / (gF**2 + g_ref**2 + eps)

    return np.mean(sim_map)


def local_variance(img, sigma=1.0):
    """Compute local variance map"""
    mu = gaussian_filter(img, sigma)
    return gaussian_filter(img**2, sigma) - mu**2


def multi_source_vif(fused, sources, sigma=1.0, eps=1e-6):
    """
    Multi-source Visual Information Fidelity.
    
    Args:
        fused: HxW fused image
        sources: list of HxW source images
        
    Returns:
        Scalar VIF score
    """
    varF = local_variance(fused, sigma)
    varS = [local_variance(s, sigma) for s in sources]

    # local information ratio per source
    info_ratios = [
        np.log2(1 + varF / (v + eps)) for v in varS
    ]

    # select dominant information contributor per pixel
    info_map = np.max(np.stack(info_ratios, axis=0), axis=0)

    return np.mean(info_map)


def calculate_multi_source_metrics(fused_img, source_imgs):
    """
    Calculate all multi-source metrics for a given fused image and its sources.
    Uses RGB channels and averages the metrics across channels.
    
    Args:
        fused_img: HxWx3 fused image (uint8)
        source_imgs: list of HxWx3 source images (uint8)
        
    Returns:
        Dictionary with metric scores
    """
    # Convert to float32 and normalize to [0,1] range
    fused = fused_img.astype(np.float32) / 255.0
    sources = [img.astype(np.float32) / 255.0 for img in source_imgs]
    
    # Initialize metrics for each channel
    mef_channels = []
    lgir_channels = []
    vif_channels = []
    
    # Process each RGB channel
    for c in range(3):  # R, G, B channels
        fused_channel = fused[:,:,c]
        sources_channels = [src[:,:,c] for src in sources]
        
        # Calculate metrics for this channel
        mef_channels.append(mef_ssim(fused_channel, sources_channels))
        lgir_channels.append(lgir_gradient_score(fused_channel, sources_channels))
        vif_channels.append(multi_source_vif(fused_channel, sources_channels))
    
    # Average metrics across all RGB channels
    mef = np.mean(mef_channels)
    lgir = np.mean(lgir_channels)
    vif = np.mean(vif_channels)
    
    return {
        "MEF_SSIM": mef,
        "LGIR": lgir,
        "MS_VIF": vif
    }
