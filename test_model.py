"""
Test if the trained model is working correctly
"""
import torch
import torch.nn.functional as F
from pathlib import Path

print("=" * 60)
print("MODEL DIAGNOSTIC TEST")
print("=" * 60)

# Check if weights exist
weights_path = Path('backend/weights/efficientnet_b4_best.pth')
print(f"\n1. Checking weights file...")
print(f"   Path: {weights_path}")
print(f"   Exists: {weights_path.exists()}")

if not weights_path.exists():
    print("\n❌ CRITICAL: Model weights not found!")
    print("   The model cannot make predictions without trained weights.")
    exit(1)

# Load checkpoint
print(f"\n2. Loading checkpoint...")
try:
    checkpoint = torch.load(weights_path, map_location='cpu')
    print(f"✓ Checkpoint loaded")
    
    # Check what's in the checkpoint
    print(f"\n3. Checkpoint contents:")
    for key in checkpoint.keys():
        if key != 'model_state_dict':
            print(f"   {key}: {checkpoint[key]}")
    
    # Check model state
    state_dict = checkpoint['model_state_dict']
    print(f"\n4. Model architecture:")
    print(f"   Total parameters: {len(state_dict)} layers")
    
    # Check for classifier layers
    classifier_layers = [k for k in state_dict.keys() if 'classifier' in k]
    print(f"   Classifier layers: {len(classifier_layers)}")
    
    for layer in classifier_layers[:5]:  # Show first 5
        print(f"      {layer}: {state_dict[layer].shape}")
    
    # Check final layer
    final_layer_keys = [k for k in state_dict.keys() if 'classifier' in k and 'weight' in k]
    if final_layer_keys:
        last_layer = final_layer_keys[-1]
        last_weight = state_dict[last_layer]
        print(f"\n5. Final layer check:")
        print(f"   Layer: {last_layer}")
        print(f"   Shape: {last_weight.shape}")
        print(f"   Output classes: {last_weight.shape[0]}")
        
        if last_weight.shape[0] == 2:
            print(f"   ✓ Correct: 2 classes (Real vs Fake)")
        else:
            print(f"   ❌ ERROR: Expected 2 classes, got {last_weight.shape[0]}")
    
    print("\n6. Training metrics:")
    val_acc = checkpoint.get('val_acc', 'N/A')
    epoch = checkpoint.get('epoch', 'N/A')
    print(f"   Validation Accuracy: {val_acc}")
    print(f"   Epoch: {epoch}")
    
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
