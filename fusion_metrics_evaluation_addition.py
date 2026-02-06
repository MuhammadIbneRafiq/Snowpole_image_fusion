"""
Cell to add to fusion_metrics_evaluation.ipynb that integrates the new multi-source metrics.
Add this code as a new cell after the main metrics calculation cell.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# Define all 25 fusion methods (same as train_all_fusion_modalities.ipynb)
FUSION_METHODS = [
    # Basic fusion methods
    "avg",
    "max", 
    "min",
    "wavg",
    "pca",
    "laplacian",
    "4scale",
    "anisotropic",
    "fpde",
    "mgf",
    "wavelet",
    # Adaptive fusion
    "adaptive",
    # CLAHE variants
    "clahe_pre",
    "clahe_post",
    # Custom weighted average variants
    "custom_wavg_1",
    "custom_wavg_2_0.5S_0.4R_0.1N",
    "custom_wavg_3_0.8S_0.1R_0.1N",
    "custom_wavg_3_pre_clahe",
    "custom_wavg_3_post_clahe",
    "custom_wavg_4_0.8N",
    "custom_wavg_5_0.8R",
    # Individual modalities
    "signal",
    "reflec",
    "nearir",
    # Combination
    "combination_4"
]

# Dataset mapping (same as train_all_fusion_modalities.ipynb)
DATASET_MAPPING = {
    "avg": "fused_avg",
    "max": "fused_max",
    "min": "fused_min",
    "wavg": "fused_wavg",
    "pca": "fused_pca",
    "laplacian": "fused_laplacian",
    "4scale": "fused_4scale",
    "anisotropic": "fused_anisotropic",
    "fpde": "fused_fpde",
    "mgf": "fused_mgf",
    "wavelet": "fused_wavelet",
    "adaptive": "adaptive_fusion_dataset",
    "clahe_pre": "clahe_pre_fused_dataset",
    "clahe_post": "clahe_post_fused_dataset",
    "custom_wavg_1": "custom_wavg_fusion_dataset",
    "custom_wavg_2_0.5S_0.4R_0.1N": "custom_wavg_0.5S_0.4R_0.1N",
    "custom_wavg_3_0.8S_0.1R_0.1N": "custom_wavg2_0.8S_0.1R_0.1N",
    "custom_wavg_3_pre_clahe": "custom_wavg3_0.8S_0.1R_0.1N_pre_clahe",
    "custom_wavg_3_post_clahe": "custom_wavg3_0.8S_0.1R_0.1N_post_clahe",
    "custom_wavg_4_0.8N": "custom_wavg4_0.1S_0.1R_0.8N",
    "custom_wavg_5_0.8R": "custom_wavg5_0.8R",
    "signal": "signal",
    "reflec": "reflec",
    "nearir": "nearir",
    "combination_4": "Combination4"
}

# Base dataset path - CORRECTED PATH for this notebook location
BASE_PATH = Path("./Effect-of-Image-Fusion-on-Snowpole-Detection/SnowPole_Detection_Dataset")
print(f"Total fusion methods: {len(FUSION_METHODS)}")
print(f"Base path: {BASE_PATH}")
print(f"Base path exists: {BASE_PATH.exists()}")

# Import the multi-source metrics module
from multi_source_metrics import calculate_multi_source_metrics

# Copy the load_image_uint8 function from the notebook
def load_image_uint8(img_path):
    """Load image and convert to uint8 RGB numpy array"""
    try:
        img = Image.open(img_path)
        img_array = np.array(img)
        
        # Handle grayscale images (convert to RGB)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array, img_array, img_array], axis=-1)
        elif img_array.shape[2] > 3:  # More than 3 channels (e.g., RGBA)
            img_array = img_array[:, :, :3]  # Keep only RGB
        
        # Ensure uint8 format
        if img_array.dtype != np.uint8:
            if np.issubdtype(img_array.dtype, np.integer):
                mx = float(np.iinfo(img_array.dtype).max)
                img_array = np.clip(
                    np.rint(img_array.astype(np.float32) / mx * 255.0),
                    0, 255
                ).astype(np.uint8)
            else:
                img_array = np.clip(
                    np.rint(img_array.astype(np.float32)),
                    0, 255
                ).astype(np.uint8)
        
        return img_array
    
    except Exception as e:
        print(f"Error loading image {img_path}: {str(e)}")
        return None

# Copy the objective_fusion_perform_fn_rgb function from the notebook
def objective_fusion_perform_fn_rgb(fused_img, source_imgs):
    """
    Calculate QABF, LABF, NABF, NABF1 metrics using all three source modalities
    and all RGB channels, averaging across all combinations.
    
    Implements unity constraint normalization (Shreyamsha Kumar, 2015):
    QABF_norm + LABF_norm + NABF_norm = 1
    
    This provides interpretable probabilistic decomposition:
    - QABF_norm: fraction of edge information preserved
    - LABF_norm: fraction of edge information lost
    - NABF_norm: fraction representing noise/artifacts introduced
    
    Args:
        fused_img: HxWx3 fused image (uint8)
        source_imgs: list of HxWx3 source images (uint8) [signal, reflec, nearir]
        
    Returns:
        tuple: (qabf, labf, nabf, nabf1, qabf_norm, labf_norm, nabf_norm)
    """
    signal_img, reflec_img, nearir_img = source_imgs
    channel_metrics = []
    
    # List of source pairs to compare with fused image
    # Each source pair is (source1, source2)
    source_pairs = [
        (signal_img, reflec_img),   # Signal + Reflec
        (signal_img, nearir_img),   # Signal + Near-IR
        (reflec_img, nearir_img)    # Reflec + Near-IR
    ]
    
    # Process each RGB channel separately
    for c in range(3):  # R, G, B channels
        channel_metrics_per_pair = []
        
        for src1, src2 in source_pairs:
            # Extract single channel for each image
            fused_channel = fused_img[:,:,c]
            src1_channel = src1[:,:,c]
            src2_channel = src2[:,:,c]
            
            # Calculate metrics for this channel and source pair
            # Using the vectorized single channel function from the notebook
            qabf, labf, nabf, nabf1, qabf_norm, labf_norm, nabf_norm = objective_fusion_perform_fn_single_channel(
                fused_channel, src1_channel, src2_channel
            )
            
            channel_metrics_per_pair.append((qabf, labf, nabf, nabf1, qabf_norm, labf_norm, nabf_norm))
        
        # Average metrics across all source pairs for this channel
        channel_qabf = np.mean([m[0] for m in channel_metrics_per_pair])
        channel_labf = np.mean([m[1] for m in channel_metrics_per_pair])
        channel_nabf = np.mean([m[2] for m in channel_metrics_per_pair])
        channel_nabf1 = np.mean([m[3] for m in channel_metrics_per_pair])
        channel_qabf_norm = np.mean([m[4] for m in channel_metrics_per_pair])
        channel_labf_norm = np.mean([m[5] for m in channel_metrics_per_pair])
        channel_nabf_norm = np.mean([m[6] for m in channel_metrics_per_pair])
        
        channel_metrics.append((channel_qabf, channel_labf, channel_nabf, channel_nabf1,
                                channel_qabf_norm, channel_labf_norm, channel_nabf_norm))
    
    # Average metrics across all RGB channels
    qabf = np.mean([m[0] for m in channel_metrics])
    labf = np.mean([m[1] for m in channel_metrics])
    nabf = np.mean([m[2] for m in channel_metrics])
    nabf1 = np.mean([m[3] for m in channel_metrics])
    qabf_norm = np.mean([m[4] for m in channel_metrics])
    labf_norm = np.mean([m[5] for m in channel_metrics])
    nabf_norm = np.mean([m[6] for m in channel_metrics])
    
    return qabf, labf, nabf, nabf1, qabf_norm, labf_norm, nabf_norm

# Copy the vectorized single channel function from the notebook
def objective_fusion_perform_fn_single_channel(fused_img, source_img_A, source_img_B):
    """
    Vectorized Petrovic fusion metrics for single channel
    """
    # Parameters for Petrovic Metrics Computation
    Td = 2
    wt_min = 0.001
    P = 1
    Lg = 1.5
    Nrg = 0.9999
    kg = 19
    sigmag = 0.5
    Nra = 0.9995
    ka = 22
    sigmaa = 0.5
    
    # Edge Strength & Orientation (vectorized)
    def sobel_vectorized(img):
        # Use numpy's gradient for vectorized computation
        gy, gx = np.gradient(img.astype(np.float32))
        return gx, gy
    
    gvA, ghA = sobel_vectorized(source_img_A)
    gA = np.sqrt(ghA**2 + gvA**2)
    
    gvB, ghB = sobel_vectorized(source_img_B)
    gB = np.sqrt(ghB**2 + gvB**2)
    
    gvF, ghF = sobel_vectorized(fused_img)
    gF = np.sqrt(ghF**2 + gvF**2)
    
    # VECTORIZED Edge Strength Ratio
    mask_AF_zero = (gA == 0) | (gF == 0)
    mask_BF_zero = (gB == 0) | (gF == 0)
    
    gAF = np.zeros_like(gF)
    gAF[~mask_AF_zero] = np.where(
        gA[~mask_AF_zero] > gF[~mask_AF_zero],
        gF[~mask_AF_zero] / gA[~mask_AF_zero],
        gA[~mask_AF_zero] / gF[~mask_AF_zero]
    )
    
    gBF = np.zeros_like(gF)
    gBF[~mask_BF_zero] = np.where(
        gB[~mask_BF_zero] > gF[~mask_BF_zero],
        gF[~mask_BF_zero] / gB[~mask_BF_zero],
        gB[~mask_BF_zero] / gF[~mask_BF_zero]
    )
    
    # VECTORIZED Edge Orientation
    mask_A_zero = (gvA == 0) & (ghA == 0)
    mask_B_zero = (gvB == 0) & (ghB == 0)
    mask_F_zero = (gvF == 0) & (ghF == 0)
    
    aA = np.arctan2(gvA, ghA)
    aA[mask_A_zero] = 0
    
    aB = np.arctan2(gvB, ghB)
    aB[mask_B_zero] = 0
    
    aF = np.arctan2(gvF, ghF)
    aF[mask_F_zero] = 0
    
    # Relative Orientation
    aAF = np.abs(np.abs(aA - aF) - np.pi / 2) * 2 / np.pi
    aBF = np.abs(np.abs(aB - aF) - np.pi / 2) * 2 / np.pi
    
    # Edge Preservation Coefficient
    QgAF = Nrg / (1 + np.exp(-kg * (gAF - sigmag)))
    QaAF = Nra / (1 + np.exp(-ka * (aAF - sigmaa)))
    QAF = np.sqrt(QgAF * QaAF)
    
    QgBF = Nrg / (1 + np.exp(-kg * (gBF - sigmag)))
    QaBF = Nra / (1 + np.exp(-ka * (aBF - sigmaa)))
    QBF = np.sqrt(QgBF * QaBF)
    
    # VECTORIZED Weights calculation
    wtA = np.where(gA >= Td, gA ** Lg, wt_min)
    wtB = np.where(gB >= Td, gB ** Lg, wt_min)
    
    wt_sum = np.sum(wtA + wtB)
    
    # VECTORIZED Quality metrics
    QAF_wt = QAF * wtA
    QBF_wt = QBF * wtB
    
    QABF = np.sum(QAF_wt + QBF_wt) / wt_sum
    LABF = np.sum((1 - QAF) * wtA + (1 - QBF) * wtB) / wt_sum
    
    # NABF calculation (vectorized)
    QAF_QBF = QAF * QBF
    QAF_QBF_wt = QAF_QBF * np.sqrt(wtA * wtB)
    NABF1 = np.sum((1 - QAF_QBF) * np.sqrt(wtA * wtB)) / np.sum(np.sqrt(wtA * wtB))
    
    # Modified NABF calculation
    rr = np.abs(gA - gB) / (gA + gB + 1e-8)
    na1 = np.abs(aA - aB) / (np.pi / 2 + 1e-8)
    na = np.sqrt(rr**2 + na1**2)
    
    mask_na_zero = (na == 0)
    na[mask_na_zero] = 1e-8
    
    QAF_QBF_na = QAF_QBF * na
    QAF_QBF_na_wt = QAF_QBF_na * np.sqrt(wtA * wtB)
    NABF = np.sum((1 - QAF_QBF_na) * np.sqrt(wtA * wtB)) / np.sum(np.sqrt(wtA * wtB))
    
    # Apply unity constraint normalization (Shreyamsha Kumar formulation)
    # QABF + LABF + NABF = 1 for interpretable probabilistic decomposition
    total = QABF + LABF + NABF
    if total > 0:
        QABF_norm = QABF / total
        LABF_norm = LABF / total
        NABF_norm = NABF / total
    else:
        QABF_norm = LABF_norm = NABF_norm = 0.0
    
    return QABF, LABF, NABF, NABF1, QABF_norm, LABF_norm, NABF_norm

def calculate_metrics_with_multi_source(method, dataset_folder, sample_images=None):
    """
    Extended version of calculate_metrics_for_fusion that also includes multi-source metrics
    """
    # Paths
    fused_path = BASE_PATH / dataset_folder / "images" / "test"
    signal_path = BASE_PATH / "signal" / "images" / "test"
    reflec_path = BASE_PATH / "reflec" / "images" / "test"
    nearir_path = BASE_PATH / "nearir" / "images" / "test"
    
    print(nearir_path, 'here it is')
    if not nearir_path.exists():
        print("NearIR path does not exist")
    
    if not reflec_path.exists():
        print("Reflectance path does not exist")

    print(fused_path, 'here is it')
    # Check if paths exist
    if not fused_path.exists():
        return {
            'method': method,
            'status': 'MISSING',
            'QABF': None, 'LABF': None, 'NABF': None, 'NABF1': None,
            'MEF_SSIM': None, 'LGIR': None, 'MS_VIF': None
        }
    
    # Get list of fused images
    all_fused_images = sorted(list(fused_path.glob("*.png")))
    
    if sample_images is not None:
        # Use specified number of images (for testing)
        fused_images = all_fused_images[:sample_images]
    else:
        # Use ALL images in test directory
        fused_images = all_fused_images
    
    if len(fused_images) == 0:
        return {
            'method': method,
            'status': 'NO_IMAGES',
            'QABF': None, 'LABF': None, 'NABF': None, 'NABF1': None,
            'MEF_SSIM': None, 'LGIR': None, 'MS_VIF': None
        }
    
    print(f"  📁 Processing {len(fused_images)} images from {fused_path}")
    
    # Initialize traditional metrics
    qabf_scores = []
    labf_scores = []
    nabf_scores = []
    nabf1_scores = []
    qabf_norm_scores = []
    labf_norm_scores = []
    nabf_norm_scores = []
    
    # Initialize multi-source metrics
    mef_ssim_scores = []
    lgir_scores = []
    ms_vif_scores = []
    
    for fused_img_path in fused_images:
        img_name = fused_img_path.name
        
        # Load corresponding source images
        signal_img_path = signal_path / img_name
        reflec_img_path = reflec_path / img_name
        nearir_img_path = nearir_path / img_name
        
        # Check if source images exist
        if not signal_img_path.exists() or not reflec_img_path.exists() or not nearir_img_path.exists():
            print(f"  ⚠️  Missing source image for {img_name}")
            continue
        
        try:
            # Load images as RGB uint8 (using existing function)
            fused_img = load_image_uint8(fused_img_path)
            signal_img = load_image_uint8(signal_img_path)
            reflec_img = load_image_uint8(reflec_img_path)
            nearir_img = load_image_uint8(nearir_img_path)
            
            if fused_img is None or signal_img is None or reflec_img is None or nearir_img is None:
                continue
            
            # Calculate traditional pairwise metrics using all three source modalities
            qabf, labf, nabf, nabf1, qabf_norm, labf_norm, nabf_norm = objective_fusion_perform_fn_rgb(
                fused_img, [signal_img, reflec_img, nearir_img]
            )
            
            qabf_scores.append(qabf)
            labf_scores.append(labf)
            nabf_scores.append(nabf)
            nabf1_scores.append(nabf1)
            qabf_norm_scores.append(qabf_norm)
            labf_norm_scores.append(labf_norm)
            nabf_norm_scores.append(nabf_norm)
            
            # Calculate new multi-source metrics using all three source modalities at once
            multi_source_results = calculate_multi_source_metrics(
                fused_img, [signal_img, reflec_img, nearir_img]
            )
            
            mef_ssim_scores.append(multi_source_results["MEF_SSIM"])
            lgir_scores.append(multi_source_results["LGIR"])
            ms_vif_scores.append(multi_source_results["MS_VIF"])
            
        except Exception as e:
            print(f"  ⚠️  Error processing {img_name}: {str(e)}")
    
    if len(qabf_scores) == 0:
        return {
            'method': method,
            'status': 'ERROR',
            'QABF': None, 'LABF': None, 'NABF': None, 'NABF1': None,
            'MEF_SSIM': None, 'LGIR': None, 'MS_VIF': None
        }
    
    # Calculate averages and standard deviations for traditional metrics
    avg_qabf = np.mean(qabf_scores)
    avg_labf = np.mean(labf_scores)
    avg_nabf = np.mean(nabf_scores)
    avg_nabf1 = np.mean(nabf1_scores)
    avg_qabf_norm = np.mean(qabf_norm_scores)
    avg_labf_norm = np.mean(labf_norm_scores)
    avg_nabf_norm = np.mean(nabf_norm_scores)
    
    std_qabf = np.std(qabf_scores)
    std_labf = np.std(labf_scores)
    std_nabf = np.std(nabf_scores)
    std_nabf1 = np.std(nabf1_scores)
    std_qabf_norm = np.std(qabf_norm_scores)
    std_labf_norm = np.std(labf_norm_scores)
    std_nabf_norm = np.std(nabf_norm_scores)
    
    # Calculate averages and standard deviations for multi-source metrics
    avg_mef_ssim = np.mean(mef_ssim_scores)
    avg_lgir = np.mean(lgir_scores)
    avg_ms_vif = np.mean(ms_vif_scores)
    
    std_mef_ssim = np.std(mef_ssim_scores)
    std_lgir = np.std(lgir_scores)
    std_ms_vif = np.std(ms_vif_scores)
    
    print(f"  ✅ QABF: {avg_qabf_norm:.4f}, LABF: {avg_labf_norm:.4f}, NABF: {avg_nabf_norm:.4f}, NABF1: {avg_nabf1:.4f}")
    print(f"     Sum (should be 1.0): {avg_qabf_norm + avg_labf_norm + avg_nabf_norm:.4f}")
    print(f"     MEF_SSIM: {avg_mef_ssim:.4f}, LGIR: {avg_lgir:.4f}, MS_VIF: {avg_ms_vif:.4f}")
    print(f"     (Calculated from {len(qabf_scores)} images)")
    
    return {
        'method': method,
        'status': 'SUCCESS',
        'num_images': len(qabf_scores),
        'QABF': avg_qabf_norm, 
        'LABF': avg_labf_norm, 
        'NABF': avg_nabf_norm, 
        'NABF1': avg_nabf1,
        'QABF_raw': avg_qabf,
        'LABF_raw': avg_labf,
        'NABF_raw': avg_nabf,
        'QABF_std': std_qabf_norm,
        'LABF_std': std_labf_norm,
        'NABF_std': std_nabf_norm,
        'NABF1_std': std_nabf1,
        'QABF_raw_std': std_qabf,
        'LABF_raw_std': std_labf,
        'NABF_raw_std': std_nabf,
        'MEF_SSIM': avg_mef_ssim,
        'LGIR': avg_lgir,
        'MS_VIF': avg_ms_vif,
        'MEF_SSIM_std': std_mef_ssim,
        'LGIR_std': std_lgir,
        'MS_VIF_std': std_ms_vif
    }

# Update the main calculation loop to use the new function
print("Computing fusion metrics for all methods with additional multi-source metrics...")
print("=" * 80)

all_results = []

for idx, method in enumerate(FUSION_METHODS, 1):
    print(f"\n[{idx}/{len(FUSION_METHODS)}] Processing: {method}")
    
    # Get dataset folder for this method
    dataset_folder = DATASET_MAPPING.get(method)
    
    if dataset_folder is None:
        print(f"⚠️  No dataset mapping found for {method}")
        continue
    
    # Calculate metrics for all test images
    result = calculate_metrics_with_multi_source(method, dataset_folder, sample_images=None)
    
    all_results.append(result)

# Create DataFrame from results
results_df = pd.DataFrame(all_results)

# Save results to CSV
csv_path = "fusion_metrics_all_methods_with_multisource.csv"
results_df.to_csv(csv_path, index=False)
print(f"\n✅ Results saved to {csv_path}")

# Display results table
results_df[['method', 'status', 'num_images', 'QABF', 'LABF', 'NABF', 'NABF1', 
            'MEF_SSIM', 'LGIR', 'MS_VIF']]
