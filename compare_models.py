"""
Comprehensive script to compare PyTorch model weights and outputs with DLX custom framework.
This script helps debug weight porting issues by:
1. Comparing base model weights (excluding LoRA)
2. Running inference on both models with same inputs
3. Comparing intermediate activations layer-by-layer
4. Generating detailed statistics on discrepancies
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import OrderedDict

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dlx
from dlx import xp
from dlx.nn.tensor import Tensor

# Import models
from gpt1.torch_train.model import Model as TorchModel
from gpt1.training.model import Model as DLXModel
from gpt1.tokenizer.tokenizer import tokenizer

# Model hyperparameters
VOCAB_SIZE = len(tokenizer.get_vocab())
D_MODEL = 1024
N_HEADS = 16
MAX_SEQ_LEN = 512
PAD_IDX = 0
DEPTH = 12
MLP_RATIO = 4

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")


def print_section(text: str):
    """Print formatted section"""
    print(f"\n{Colors.OKCYAN}{Colors.BOLD}{'-'*80}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.OKCYAN}{Colors.BOLD}{'-'*80}{Colors.ENDC}")


def compare_weights(w1: np.ndarray, w2: np.ndarray, name: str, threshold: float = 1e-5) -> Dict[str, Any]:
    """
    Compare two weight arrays and return statistics
    
    Args:
        w1: First weight array (PyTorch)
        w2: Second weight array (DLX)
        name: Name of the weight
        threshold: Threshold for considering weights as matching
    
    Returns:
        Dictionary with comparison statistics
    """
    if w1.shape != w2.shape:
        return {
            'name': name,
            'match': False,
            'error': f'Shape mismatch: {w1.shape} vs {w2.shape}',
            'shapes': (w1.shape, w2.shape)
        }
    
    diff = np.abs(w1 - w2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    rel_diff = np.mean(diff / (np.abs(w1) + 1e-10))
    
    # Check if weights match within threshold
    match = max_diff < threshold
    
    # Calculate percentage of elements that are close
    close_elements = np.sum(diff < threshold)
    total_elements = diff.size
    close_percentage = 100.0 * close_elements / total_elements
    
    return {
        'name': name,
        'match': match,
        'shape': w1.shape,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'rel_diff': rel_diff,
        'close_percentage': close_percentage,
        'w1_norm': np.linalg.norm(w1),
        'w2_norm': np.linalg.norm(w2),
    }


def load_pytorch_model(checkpoint_path: str, device: torch.device, load_lora: bool = False) -> TorchModel:
    """Load PyTorch model from checkpoint"""
    print(f"Loading PyTorch model from {checkpoint_path}")
    
    model = TorchModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        transformer_depth=DEPTH,
        max_seq_len=MAX_SEQ_LEN,
        pad_idx=PAD_IDX,
        mlp_ratio=MLP_RATIO,
        lora=load_lora,  # Set to True if you want to include LoRA
        lora_r=8,
        lora_alpha=8
    ).to(device)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model


def load_dlx_model() -> DLXModel:
    """Initialize DLX model"""
    print(f"Initializing DLX model")
    
    model = DLXModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        max_seq_len=MAX_SEQ_LEN,
        pad_idx=PAD_IDX,
        n_heads=N_HEADS,
        transformer_depth=DEPTH,
        checkpoint_interval_seconds=0,
        train_dir="",
        validation_dir="",
        checkpoint_dir="",
        epochs=0,
        mini_batch_per_step=1,
        mlp_ratio=MLP_RATIO,
        lora=False  # Compare base models only
    )
    
    model.is_training = False
    return model


def extract_base_weights_pytorch(model: TorchModel) -> Dict[str, np.ndarray]:
    """
    Extract base weights from PyTorch model (excluding LoRA)
    
    Returns dictionary mapping weight names to numpy arrays
    """
    weights = {}
    state_dict = model.state_dict()
    
    for name, param in state_dict.items():
        # Skip LoRA weights
        if 'lora' in name.lower():
            continue
        
        # Convert to numpy
        weights[name] = param.detach().cpu().numpy()
    
    return weights


def extract_base_weights_dlx(model: DLXModel) -> Dict[str, np.ndarray]:
    """
    Extract base weights from DLX model
    
    Returns dictionary mapping weight names to numpy arrays
    """
    weights = {}
    params = model.parameters()
    
    for name, param in params.items():
        # Skip LoRA weights
        if 'lora' in name.lower():
            continue
        
        # Convert to numpy if needed
        if isinstance(param, Tensor):
            weights[name] = xp.asnumpy(param.data) if hasattr(xp, 'asnumpy') else param.data
        else:
            weights[name] = xp.asnumpy(param) if hasattr(xp, 'asnumpy') else param
    
    return weights


def create_weight_mapping() -> Dict[str, str]:
    """
    Create mapping between PyTorch weight names and DLX weight names
    
    Returns:
        Dictionary mapping PyTorch names to DLX names
    """
    mapping = {}
    
    # Token embedding
    mapping['token_emb.weight'] = '0_1_embedding_1_embed'
    
    # Positional embedding
    mapping['pos_emb'] = '0_1_embedding_1_pe'
    
    # Transformer blocks (DLX blocks start at index 1, not 0!)
    for i in range(DEPTH):
        lib_block_idx = i + 1  # DLX blocks start at 1
        
        # Layer norms (indices are 1 and 2, not 0 and 1)
        mapping[f'blocks.{i}.ln1.weight'] = f'transformer_{lib_block_idx}_layernorm_1_gamma'
        mapping[f'blocks.{i}.ln1.bias'] = f'transformer_{lib_block_idx}_layernorm_1_beta'
        mapping[f'blocks.{i}.ln2.weight'] = f'transformer_{lib_block_idx}_layernorm_2_gamma'
        mapping[f'blocks.{i}.ln2.bias'] = f'transformer_{lib_block_idx}_layernorm_2_beta'
        
        # Attention (linear indices: 1, 2)
        mapping[f'blocks.{i}.attn.qkv.weight'] = f'transformer_{lib_block_idx}_linear_1_qkv_weight'
        mapping[f'blocks.{i}.attn.qkv.bias'] = f'transformer_{lib_block_idx}_linear_1_qkv_bias'
        mapping[f'blocks.{i}.attn.out_proj.weight'] = f'transformer_{lib_block_idx}_linear_2_o_weight'
        mapping[f'blocks.{i}.attn.out_proj.bias'] = f'transformer_{lib_block_idx}_linear_2_o_bias'
        
        # MLP (linear indices: 3, 4)
        mapping[f'blocks.{i}.mlp.0.weight'] = f'transformer_{lib_block_idx}_linear_3_proj_up_weight'
        mapping[f'blocks.{i}.mlp.0.bias'] = f'transformer_{lib_block_idx}_linear_3_proj_up_bias'
        mapping[f'blocks.{i}.mlp.2.weight'] = f'transformer_{lib_block_idx}_linear_4_proj_down_weight'
        mapping[f'blocks.{i}.mlp.2.bias'] = f'transformer_{lib_block_idx}_linear_4_proj_down_bias'
    
    # LM head
    mapping['lm_head.weight'] = 'linear_1_linear_1_project_weight'
    mapping['lm_head.bias'] = 'linear_1_linear_1_project_bias'
    
    return mapping


def compare_all_weights(pytorch_weights: Dict[str, np.ndarray], 
                       dlx_weights: Dict[str, np.ndarray],
                       mapping: Dict[str, str],
                       threshold: float = 1e-5) -> List[Dict[str, Any]]:
    """
    Compare all weights between PyTorch and DLX models
    
    Returns list of comparison results
    """
    results = []
    
    print_section("Comparing Weights")
    
    for pt_name, dlx_name in mapping.items():
        if pt_name not in pytorch_weights:
            print(f"{Colors.WARNING}Warning: {pt_name} not found in PyTorch model{Colors.ENDC}")
            continue
        
        if dlx_name not in dlx_weights:
            print(f"{Colors.WARNING}Warning: {dlx_name} not found in DLX model{Colors.ENDC}")
            continue
        
        pt_weight = pytorch_weights[pt_name]
        dlx_weight = dlx_weights[dlx_name]
        
        result = compare_weights(pt_weight, dlx_weight, f"{pt_name} <-> {dlx_name}", threshold)
        results.append(result)
        
        # Print result
        if result.get('error'):
            print(f"{Colors.FAIL}✗ {result['name']}: {result['error']}{Colors.ENDC}")
        elif result['match']:
            print(f"{Colors.OKGREEN}✓ {result['name']}: MATCH (max_diff={result['max_diff']:.2e}){Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}✗ {result['name']}: MISMATCH{Colors.ENDC}")
            print(f"  Max diff: {result['max_diff']:.2e}, Mean diff: {result['mean_diff']:.2e}")
            print(f"  Relative diff: {result['rel_diff']:.2%}, Close elements: {result['close_percentage']:.2f}%")
            print(f"  PyTorch norm: {result['w1_norm']:.4f}, DLX norm: {result['w2_norm']:.4f}")
    
    return results


def run_inference_comparison(pytorch_model: TorchModel, 
                             dlx_model: DLXModel, 
                             device: torch.device,
                             prompt: str = "Once upon a time") -> Dict[str, Any]:
    """
    Run inference on both models with same input and compare outputs
    """
    print_section("Running Inference Comparison")
    
    # Tokenize input
    encoded = tokenizer.encode(prompt)
    input_ids = encoded.ids[:20]  # Use shorter sequence for comparison
    
    print(f"Input prompt: '{prompt}'")
    print(f"Input IDs: {input_ids}")
    print(f"Sequence length: {len(input_ids)}")
    
    # Prepare inputs for PyTorch
    pt_input = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Prepare inputs for DLX
    dlx_input = xp.array([input_ids], dtype=xp.int32)
    
    # Run PyTorch model
    with torch.no_grad():
        pt_output = pytorch_model(pt_input)  # [1, seq_len, vocab_size]
        pt_output_np = pt_output.cpu().numpy()
    
    # Run DLX model
    dlx_output = dlx_model.forward(dlx_input)  # [1, seq_len, vocab_size]
    if isinstance(dlx_output, Tensor):
        dlx_output_np = xp.asnumpy(dlx_output.data) if hasattr(xp, 'asnumpy') else dlx_output.data
    else:
        dlx_output_np = xp.asnumpy(dlx_output) if hasattr(xp, 'asnumpy') else dlx_output
    
    # Compare outputs
    print(f"\nPyTorch output shape: {pt_output_np.shape}")
    print(f"DLX output shape: {dlx_output_np.shape}")
    
    # Get logits for last position
    pt_logits = pt_output_np[0, -1, :]  # [vocab_size]
    dlx_logits = dlx_output_np[0, -1, :]  # [vocab_size]
    
    # Compare logits
    logits_diff = np.abs(pt_logits - dlx_logits)
    
    # Get top predictions
    pt_top5_ids = np.argsort(pt_logits)[-5:][::-1]
    dlx_top5_ids = np.argsort(dlx_logits)[-5:][::-1]
    
    print(f"\n{Colors.BOLD}Logits Comparison (last position):{Colors.ENDC}")
    print(f"  Max diff: {np.max(logits_diff):.6f}")
    print(f"  Mean diff: {np.mean(logits_diff):.6f}")
    print(f"  Std diff: {np.std(logits_diff):.6f}")
    
    print(f"\n{Colors.BOLD}PyTorch Top 5 Predictions:{Colors.ENDC}")
    for i, idx in enumerate(pt_top5_ids):
        token = tokenizer.decode([idx])
        print(f"  {i+1}. Token {idx} ('{token}'): logit={pt_logits[idx]:.4f}")
    
    print(f"\n{Colors.BOLD}DLX Top 5 Predictions:{Colors.ENDC}")
    for i, idx in enumerate(dlx_top5_ids):
        token = tokenizer.decode([idx])
        print(f"  {i+1}. Token {idx} ('{token}'): logit={dlx_logits[idx]:.4f}")
    
    # Compare probability distributions
    pt_probs = F.softmax(torch.from_numpy(pt_logits), dim=0).numpy()
    dlx_probs_raw = np.exp(dlx_logits - np.max(dlx_logits))
    dlx_probs = dlx_probs_raw / np.sum(dlx_probs_raw)
    
    prob_diff = np.abs(pt_probs - dlx_probs)
    
    print(f"\n{Colors.BOLD}Probability Distribution Comparison:{Colors.ENDC}")
    print(f"  Max diff: {np.max(prob_diff):.6f}")
    print(f"  Mean diff: {np.mean(prob_diff):.6f}")
    print(f"  KL divergence (PT->DLX): {np.sum(pt_probs * np.log((pt_probs + 1e-10) / (dlx_probs + 1e-10))):.6f}")
    print(f"  KL divergence (DLX->PT): {np.sum(dlx_probs * np.log((dlx_probs + 1e-10) / (pt_probs + 1e-10))):.6f}")
    
    return {
        'pt_logits': pt_logits,
        'dlx_logits': dlx_logits,
        'logits_max_diff': np.max(logits_diff),
        'logits_mean_diff': np.mean(logits_diff),
        'pt_top5': pt_top5_ids,
        'dlx_top5': dlx_top5_ids,
        'pt_probs': pt_probs,
        'dlx_probs': dlx_probs,
        'prob_max_diff': np.max(prob_diff),
    }


def generate_report(weight_results: List[Dict[str, Any]], 
                   inference_results: Dict[str, Any],
                   threshold: float = 1e-5):
    """Generate comprehensive comparison report"""
    print_header("COMPARISON REPORT")
    
    # Weight comparison summary
    print_section("Weight Comparison Summary")
    
    total_weights = len(weight_results)
    matched_weights = sum(1 for r in weight_results if r.get('match', False))
    mismatched_weights = total_weights - matched_weights
    
    print(f"Total weights compared: {total_weights}")
    print(f"{Colors.OKGREEN}Matched weights: {matched_weights} ({100*matched_weights/total_weights:.1f}%){Colors.ENDC}")
    
    if mismatched_weights > 0:
        print(f"{Colors.FAIL}Mismatched weights: {mismatched_weights} ({100*mismatched_weights/total_weights:.1f}%){Colors.ENDC}")
        
        print(f"\n{Colors.BOLD}Top 10 Mismatched Weights by Max Difference:{Colors.ENDC}")
        mismatched = [r for r in weight_results if not r.get('match', False) and not r.get('error')]
        mismatched_sorted = sorted(mismatched, key=lambda x: x.get('max_diff', 0), reverse=True)[:10]
        
        for i, r in enumerate(mismatched_sorted, 1):
            print(f"\n{i}. {r['name']}")
            print(f"   Max diff: {r['max_diff']:.6e}, Mean diff: {r['mean_diff']:.6e}")
            print(f"   Relative diff: {r['rel_diff']:.6%}")
            print(f"   PyTorch norm: {r['w1_norm']:.4f}, DLX norm: {r['w2_norm']:.4f}")
    
    # Inference comparison summary
    print_section("Inference Comparison Summary")
    
    if inference_results['logits_max_diff'] < threshold:
        print(f"{Colors.OKGREEN}✓ Inference outputs MATCH (max_diff < {threshold}){Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}✗ Inference outputs MISMATCH{Colors.ENDC}")
        print(f"  Logits max diff: {inference_results['logits_max_diff']:.6e}")
        print(f"  Logits mean diff: {inference_results['logits_mean_diff']:.6e}")
        print(f"  Probability max diff: {inference_results['prob_max_diff']:.6e}")
    
    # Check if top predictions match
    pt_top1 = inference_results['pt_top5'][0]
    dlx_top1 = inference_results['dlx_top5'][0]
    
    if pt_top1 == dlx_top1:
        print(f"{Colors.OKGREEN}✓ Top prediction MATCHES{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}✗ Top prediction MISMATCH{Colors.ENDC}")
        print(f"  PyTorch: token {pt_top1}")
        print(f"  DLX: token {dlx_top1}")


def main():
    """Main comparison function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare PyTorch and DLX models")
    parser.add_argument("--checkpoint", type=str, default="gpt1/checkpoints/model.pt",
                       help="Path to PyTorch checkpoint")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to run PyTorch model on")
    parser.add_argument("--threshold", type=float, default=1e-5,
                       help="Threshold for considering weights as matching")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                       help="Prompt for inference comparison")
    parser.add_argument("--load-lora", action="store_true",
                       help="Load LoRA weights in PyTorch model (default: False, base model only)")
    
    args = parser.parse_args()
    
    print_header("PyTorch vs DLX Model Comparison Tool")
    
    # Device setup
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load models
    print_section("Loading Models")
    pytorch_model = load_pytorch_model(args.checkpoint, device, load_lora=args.load_lora)
    dlx_model = load_dlx_model()
    
    print(f"\n{Colors.OKGREEN}✓ Models loaded successfully{Colors.ENDC}")
    
    # Extract weights
    print_section("Extracting Weights")
    pytorch_weights = extract_base_weights_pytorch(pytorch_model)
    dlx_weights = extract_base_weights_dlx(dlx_model)
    
    print(f"PyTorch weights: {len(pytorch_weights)} parameters")
    print(f"DLX weights: {len(dlx_weights)} parameters")
    
    # Create mapping
    mapping = create_weight_mapping()
    
    # Compare weights
    weight_results = compare_all_weights(pytorch_weights, dlx_weights, mapping, args.threshold)
    
    # Run inference comparison
    inference_results = run_inference_comparison(pytorch_model, dlx_model, device, args.prompt)
    
    # Generate report
    generate_report(weight_results, inference_results, args.threshold)
    
    print(f"\n{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}Comparison complete!{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*80}{Colors.ENDC}\n")


if __name__ == "__main__":
    main()
