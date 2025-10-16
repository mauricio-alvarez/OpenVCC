import os
import argparse
import sys

CUSTOM_CACHE_DIR = "model_cache"

def main():
    """
    Parses command-line arguments and downloads a timm model to a specific directory.
    """
    # 1. Set up argument parser
    parser = argparse.ArgumentParser(
        description="Download and cache a pretrained model from the 'timm' library.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="The name of the timm model to download (e.g., 'vit_huge_patch14_224.in21k')."
    )
    
    # If no arguments are given, print help and exit
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    os.environ['TORCH_HOME'] = CUSTOM_CACHE_DIR
    os.makedirs(CUSTOM_CACHE_DIR, exist_ok=True)
    
    import timm
    
    print(f"Set TORCH_HOME to: {os.path.abspath(CUSTOM_CACHE_DIR)}")
    print(f"Attempting to download and cache '{args.model_name}'...")

    # 3. Trigger the download
    try:
        model = timm.create_model(args.model_name, pretrained=True)
        print("\n Success! Model downloaded and cached.")
        # The actual weights are stored in a subdirectory created by torch.hub
        print(f"Look for the model files inside the '{CUSTOM_CACHE_DIR}/hub/checkpoints/' directory.")
    except Exception as e:
        print(f"\n Error: Could not download the model. Please check the model name and your internet connection.")
        print(f"  Details: {e}")

if __name__ == "__main__":
    main()
