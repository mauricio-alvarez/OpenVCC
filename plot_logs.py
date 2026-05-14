import sys
import re
import matplotlib.pyplot as plt
import os

def plot_training_logs(log_file_path, output_png_path=None):
    if not os.path.exists(log_file_path):
        print(f"Error: Log file {log_file_path} not found.")
        return

    if output_png_path is None:
        base_name = os.path.splitext(os.path.basename(log_file_path))[0]
        output_png_path = f"{base_name}_curves.png"

    epochs = []
    avg_losses = []
    val_accs = []
    
    # We can also parse step losses if needed, but epoch averages are cleaner
    # For now, we'll extract the "Epoch X Done" lines
    pattern = re.compile(r"Epoch (\d+) Done\. Avg Loss: ([0-9.]+) \| Val Acc: ([0-9.]+)%")
    
    with open(log_file_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                avg_losses.append(float(match.group(2)))
                val_accs.append(float(match.group(3)))
                
    if not epochs:
        print(f"No epoch summary data found in {log_file_path}. Please check the log format.")
        return
        
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Average Loss on the left y-axis
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Loss', color=color)
    ax1.plot(epochs, avg_losses, color=color, marker='o', label='Average Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot Validation Accuracy on the right y-axis
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Validation Accuracy (%)', color=color)  
    ax2.plot(epochs, val_accs, color=color, marker='s', label='Val Acc')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and layout
    plt.title(f'Training Metrics: {os.path.basename(log_file_path)}')
    fig.tight_layout()  
    
    # Save the plot
    plt.savefig(output_png_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated plot with {len(epochs)} epochs.")
    print(f"Saved training curves to: {output_png_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_logs.py <path_to_log_file> [output_png_path]")
        sys.exit(1)
        
    log_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    plot_training_logs(log_file, out_file)
