"""
Create inference visualization figures for fusion methods
Shows source modalities and fusion results with detection boxes
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import cv2

# Base paths
BASE_PATH = Path("./Effect-of-Image-Fusion-on-Snowpole-Detection/SnowPole_Detection_Dataset")
WEIGHTS_BASE = Path("./Effect-of-Image-Fusion-on-Snowpole-Detection/runs_yolo11n_fusion")

# Dataset mapping
DATASET_MAPPING = {
    "signal": "signal",
    "reflec": "reflec",
    "nearir": "nearir",
    "combination_4": "Combination4",
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
}

# Weights path mapping
WEIGHTS_MAPPING = {
    "signal": "train_signal/weights/best.pt",
    "reflec": "train_reflec/weights/best.pt",
    "nearir": "train_nearir/weights/best.pt",
    "combination_4": "train_combination_4/weights/best.pt",
    "avg": "train_average_fusion/weights/best.pt",
    "max": "train_max_fusion/weights/best.pt",
    "min": "train_min_fusion/weights/best.pt",
    "wavg": "train_weighted_average_fusion/weights/best.pt",
    "pca": "train_pca_fusion/weights/best.pt",
    "laplacian": "train_laplacian_pyramid_fusion/weights/best.pt",
    "4scale": "train_4-scale_fusion/weights/best.pt",
    "anisotropic": "train_anisotropic_fusion/weights/best.pt",
    "fpde": "train_fpde/weights/best.pt",
    "mgf": "train_mgf/weights/best.pt",
    "wavelet": "train_wavelet_fusion/weights/best.pt",
    "adaptive": "train_adaptive_fusion/weights/best.pt",
    "clahe_pre": "train_clahe_pre/weights/best.pt",
    "clahe_post": "train_clahe_post/weights/best.pt",
    "custom_wavg_1": "custom_wavg_1/weights/best.pt",
    "custom_wavg_2_0.5S_0.4R_0.1N": "custom_wavg_2_0.5S_0.4R_0.1N/weights/best.pt",
    "custom_wavg_3_0.8S_0.1R_0.1N": "custom_wavg_3_0.8S_0.1R_0.1N/weights/best.pt",
    "custom_wavg_3_pre_clahe": "custom_wavg_3_pre_clahe/weights/best.pt",
    "custom_wavg_3_post_clahe": "custom_wavg_3_post_clahe/weights/best.pt",
    "custom_wavg_4_0.8N": "custom_wavg_4_0.8N/weights/best.pt",
    "custom_wavg_5_0.8R": "custom_wavg5_0.8R/weights/best.pt",
}


def load_and_normalize_image(img_path):
    """Load image and normalize for display"""
    img = Image.open(img_path)
    img_array = np.array(img)
    
    # Convert grayscale to RGB if needed
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array, img_array, img_array], axis=-1)
    elif img_array.shape[2] > 3:
        img_array = img_array[:, :, :3]
    
    # Normalize to 0-255 uint8
    if img_array.dtype != np.uint8:
        img_min, img_max = img_array.min(), img_array.max()
        if img_max > img_min:
            img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_array = np.zeros_like(img_array, dtype=np.uint8)
    
    return img_array


def run_inference_and_draw(img_array, weights_path, conf_threshold=0.25):
    """Run YOLO inference and draw bounding boxes"""
    # Load model
    model = YOLO(str(weights_path))
    
    # Run inference
    results = model.predict(img_array, conf=conf_threshold, verbose=False)
    
    # Draw boxes on image
    img_with_boxes = img_array.copy()
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            
            # Draw rectangle
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence
            label = f'{conf:.2f}'
            cv2.putText(img_with_boxes, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_with_boxes


def create_figure_1(test_image_name="image_0.png"):
    """
    Create Figure 1: Source modalities + basic fusion methods
    Methods: Signal, Reflec, Near-IR, Combination 4, Avg, Max, Min, Wavg, PCA, Laplacian
    """
    methods = [
        ("signal", "Signal (I;16)"),
        ("reflec", "Reflec (I;16)"),
        ("nearir", "Near-IR (I;16)"),
        ("combination_4", "Combination 4 (R=NearIR, G=Signal, B=Reflec)"),
        ("avg", "Fused Avg (I;16)"),
        ("max", "Fused Max (I;16)"),
        ("min", "Fused Min (I;16)"),
        ("wavg", "Fused Weighted Avg (I;16)"),
        ("pca", "Fused PCA (I;16)"),
        ("laplacian", "Fused Laplacian Pyramid (I;16)"),
    ]
    
    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 2.5 * n_methods))
    fig.patch.set_facecolor('white')
    
    for idx, (method_key, method_label) in enumerate(methods):
        ax = axes[idx]
        
        # Get image path
        dataset_folder = DATASET_MAPPING[method_key]
        img_path = BASE_PATH / dataset_folder / "images" / "test" / test_image_name
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            ax.text(0.5, 0.5, f"Image not found\n{img_path}", 
                   ha='center', va='center', fontsize=10)
            ax.axis('off')
            continue
        
        # Load image
        img_array = load_and_normalize_image(img_path)
        
        # Run inference
        weights_path = WEIGHTS_BASE / WEIGHTS_MAPPING[method_key]
        if weights_path.exists():
            img_with_boxes = run_inference_and_draw(img_array, weights_path)
        else:
            print(f"Warning: Weights not found: {weights_path}")
            img_with_boxes = img_array
        
        # Display
        ax.imshow(img_with_boxes, cmap='gray' if len(img_with_boxes.shape) == 2 else None)
        ax.set_title(method_label, fontsize=10, pad=5)
        ax.axis('off')
    
    plt.tight_layout()
    output_path = "figure_1_basic_fusion_inference.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_figure_2(test_image_name="image_0.png"):
    """
    Create Figure 2: Advanced fusion methods (non-weighted)
    Methods: Adaptive, 4-scale, Anisotropic, FPDE, MGF, Wavelet
    """
    methods = [
        ("adaptive", "Fused Adaptive (I;16)"),
        ("4scale", "Fused 4-scale (I;16)"),
        ("anisotropic", "Fused Anisotropic (I;16)"),
        ("fpde", "Fused FPDE (I;16)"),
        ("mgf", "Fused MGF (I;16)"),
        ("wavelet", "Fused Wavelet (I;16)"),
    ]
    
    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 2.5 * n_methods))
    fig.patch.set_facecolor('white')
    
    for idx, (method_key, method_label) in enumerate(methods):
        ax = axes[idx]
        
        # Get image path
        dataset_folder = DATASET_MAPPING[method_key]
        img_path = BASE_PATH / dataset_folder / "images" / "test" / test_image_name
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            ax.text(0.5, 0.5, f"Image not found\n{img_path}", 
                   ha='center', va='center', fontsize=10)
            ax.axis('off')
            continue
        
        # Load image
        img_array = load_and_normalize_image(img_path)
        
        # Run inference
        weights_path = WEIGHTS_BASE / WEIGHTS_MAPPING[method_key]
        if weights_path.exists():
            img_with_boxes = run_inference_and_draw(img_array, weights_path)
        else:
            print(f"Warning: Weights not found: {weights_path}")
            img_with_boxes = img_array
        
        # Display
        ax.imshow(img_with_boxes, cmap='gray' if len(img_with_boxes.shape) == 2 else None)
        ax.set_title(method_label, fontsize=10, pad=5)
        ax.axis('off')
    
    plt.tight_layout()
    output_path = "figure_2_advanced_fusion_inference.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    plt.close()


def create_figure_3(test_image_name="image_0.png"):
    """
    Create Figure 3: Weighted average and CLAHE fusion methods
    Methods: Custom wavg 1-5, Pre/Post CLAHE variants
    """
    methods = [
        ("custom_wavg_1", "Custom Weighted Avg 1 (I;16)"),
        ("custom_wavg_2_0.5S_0.4R_0.1N", "Custom Weighted Avg 2 (0.5S, 0.4R, 0.1N)"),
        ("custom_wavg_3_0.8S_0.1R_0.1N", "Custom Weighted Avg 3 (0.8S, 0.1R, 0.1N)"),
        ("custom_wavg_4_0.8N", "Custom Weighted Avg 4 (0.1S, 0.1R, 0.8N)"),
        ("custom_wavg_5_0.8R", "Custom Weighted Avg 5 (0.1S, 0.8R, 0.1N)"),
        ("clahe_pre", "Pre-CLAHE Fusion (I;16)"),
        ("clahe_post", "Post-CLAHE Fusion (I;16)"),
        ("custom_wavg_3_pre_clahe", "Pre-CLAHE Custom Weighted Avg 3"),
        ("custom_wavg_3_post_clahe", "Post-CLAHE Custom Weighted Avg 3"),
    ]
    
    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 1, figsize=(12, 2.5 * n_methods))
    fig.patch.set_facecolor('white')
    
    for idx, (method_key, method_label) in enumerate(methods):
        ax = axes[idx]
        
        # Get image path
        dataset_folder = DATASET_MAPPING[method_key]
        img_path = BASE_PATH / dataset_folder / "images" / "test" / test_image_name
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            ax.text(0.5, 0.5, f"Image not found\n{img_path}", 
                   ha='center', va='center', fontsize=10)
            ax.axis('off')
            continue
        
        # Load image
        img_array = load_and_normalize_image(img_path)
        
        # Run inference
        weights_path = WEIGHTS_BASE / WEIGHTS_MAPPING[method_key]
        if weights_path.exists():
            img_with_boxes = run_inference_and_draw(img_array, weights_path)
        else:
            print(f"Warning: Weights not found: {weights_path}")
            img_with_boxes = img_array
        
        # Display
        ax.imshow(img_with_boxes, cmap='gray' if len(img_with_boxes.shape) == 2 else None)
        ax.set_title(method_label, fontsize=10, pad=5)
        ax.axis('off')
    
    plt.tight_layout()
    output_path = "figure_3_weighted_clahe_inference.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("Creating inference visualization figures...")
    print("=" * 80)
    
    # Select a test image (you can change this)
    test_image = "image_0.png"
    
    print(f"\nUsing test image: {test_image}")
    print("\nCreating Figure 1: Basic fusion methods with inference...")
    create_figure_1(test_image)
    
    print("\nCreating Figure 2: Advanced fusion methods with inference...")
    create_figure_2(test_image)
    
    print("\nCreating Figure 3: Weighted average and CLAHE fusion methods with inference...")
    create_figure_3(test_image)
    
    print("\n" + "=" * 80)
    print("✅ All 3 figures created successfully!")
    print("\nOutput files:")
    print("  - figure_1_basic_fusion_inference.png")
    print("  - figure_2_advanced_fusion_inference.png")
    print("  - figure_3_weighted_clahe_inference.png")
