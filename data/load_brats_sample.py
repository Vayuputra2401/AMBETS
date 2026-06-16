"""
BraTS2021 Data Loader and Visualizer

This script loads a single BraTS2021 sample and provides comprehensive visualization
including MRI slices, segmentation masks, bounding boxes, and data characteristics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

class BraTSDataLoader:
    def __init__(self, data_root, sample_name):
        """
        Initialize BraTS data loader
        
        Args:
            data_root (str): Root directory of BraTS dataset
            sample_name (str): Name of the sample (e.g., 'BraTS-GLI-00005-100')
        """
        self.data_root = data_root
        self.sample_name = sample_name
        self.sample_path = os.path.join(data_root, sample_name)
        
        # Define modality file patterns with correct naming
        self.modalities = {
            't1c': f'{sample_name}-t1c.nii.gz',
            't1n': f'{sample_name}-t1n.nii.gz',
            't2w': f'{sample_name}-t2w.nii.gz',
            't2f': f'{sample_name}-t2f.nii.gz',
            'seg': f'{sample_name}-seg.nii.gz'
        }
        
        self.data = {}
        self.headers = {}
        
        # Create output directory for visualizations
        self.output_dir = os.path.join("visualizations", sample_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_data(self):
        """Load all modalities and segmentation"""
        print(f"Loading BraTS sample: {self.sample_name}")
        print(f"Sample path: {self.sample_path}")
        print(f"Output directory: {self.output_dir}")
        print("-" * 60)
        
        for modality, filename in self.modalities.items():
            file_path = os.path.join(self.sample_path, filename)
            
            if os.path.exists(file_path):
                try:
                    nii_img = nib.load(file_path)
                    self.data[modality] = nii_img.get_fdata()
                    self.headers[modality] = nii_img.header
                    print(f"✓ Loaded {modality}: {filename}")
                except Exception as e:
                    print(f"✗ Failed to load {modality}: {e}")
            else:
                print(f"✗ File not found: {filename}")
        
        print("-" * 60)
        return self.data
    
    def print_characteristics(self):
        """Print detailed characteristics of the loaded data"""
        print("\n" + "="*80)
        print("DATA CHARACTERISTICS")
        print("="*80)
        
        for modality, data in self.data.items():
            print(f"\n{modality.upper()} Modality:")
            print(f"  Shape: {data.shape}")
            print(f"  Data type: {data.dtype}")
            print(f"  Min value: {np.min(data):.4f}")
            print(f"  Max value: {np.max(data):.4f}")
            print(f"  Mean value: {np.mean(data):.4f}")
            print(f"  Std value: {np.std(data):.4f}")
            
            if modality == 'seg':
                unique_labels = np.unique(data)
                print(f"  Unique labels: {unique_labels}")
                for label in unique_labels:
                    count = np.sum(data == label)
                    percentage = (count / data.size) * 100
                    print(f"    Label {int(label)}: {count} voxels ({percentage:.2f}%)")
            
            # Print header information
            if modality in self.headers:
                header = self.headers[modality]
                print(f"  Voxel size: {header.get_zooms()[:3]}")
                print(f"  Orientation: {nib.aff2axcodes(header.get_best_affine())}")
    
    def save_characteristics_to_file(self):
        """Save detailed characteristics to a text file"""
        output_file = os.path.join(self.output_dir, "data_characteristics.txt")
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"BraTS Sample: {self.sample_name}\n")
            f.write("DATA CHARACTERISTICS\n")
            f.write("="*80 + "\n")
            
            for modality, data in self.data.items():
                f.write(f"\n{modality.upper()} Modality:\n")
                f.write(f"  Shape: {data.shape}\n")
                f.write(f"  Data type: {data.dtype}\n")
                f.write(f"  Min value: {np.min(data):.4f}\n")
                f.write(f"  Max value: {np.max(data):.4f}\n")
                f.write(f"  Mean value: {np.mean(data):.4f}\n")
                f.write(f"  Std value: {np.std(data):.4f}\n")
                
                if modality == 'seg':
                    unique_labels = np.unique(data)
                    f.write(f"  Unique labels: {unique_labels}\n")
                    for label in unique_labels:
                        count = np.sum(data == label)
                        percentage = (count / data.size) * 100
                        f.write(f"    Label {int(label)}: {count} voxels ({percentage:.2f}%)\n")
                
                # Write header information
                if modality in self.headers:
                    header = self.headers[modality]
                    f.write(f"  Voxel size: {header.get_zooms()[:3]}\n")
                    f.write(f"  Orientation: {nib.aff2axcodes(header.get_best_affine())}\n")
        
        print(f"✓ Characteristics saved to: {output_file}")
    
    def get_bounding_box(self, mask, margin=5):
        """
        Calculate bounding box for non-zero regions in the mask
        
        Args:
            mask (np.array): Binary or labeled mask
            margin (int): Additional margin around the bounding box
            
        Returns:
            tuple: (min_x, max_x, min_y, max_y, min_z, max_z)
        """
        # Find non-zero regions
        coords = np.where(mask > 0)
        
        if len(coords[0]) == 0:
            return None
        
        min_x, max_x = max(0, coords[0].min() - margin), min(mask.shape[0], coords[0].max() + margin + 1)
        min_y, max_y = max(0, coords[1].min() - margin), min(mask.shape[1], coords[1].max() + margin + 1)
        min_z, max_z = max(0, coords[2].min() - margin), min(mask.shape[2], coords[2].max() + margin + 1)
        
        return (min_x, max_x, min_y, max_y, min_z, max_z)
    
    def plot_sample_overview(self, save=False):
        """Plot overview of all modalities for middle slice"""
        if not self.data:
            print("No data loaded. Please run load_data() first.")
            return
        
        # Get middle slice index
        sample_shape = list(self.data.values())[0].shape
        middle_slice = sample_shape[2] // 2
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'BraTS Sample: {self.sample_name} (Slice {middle_slice})', fontsize=16)
        
        modality_names = ['t1c', 't1n', 't2w', 't2f', 'seg']
        plot_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        
        for i, modality in enumerate(modality_names):
            if modality in self.data:
                row, col = plot_positions[i]
                ax = axes[row, col]
                
                slice_data = self.data[modality][:, :, middle_slice]
                
                if modality == 'seg':
                    # Use discrete colormap for segmentation
                    im = ax.imshow(slice_data.T, cmap='tab10', vmin=0, vmax=4, origin='lower')
                    ax.set_title(f'{modality.upper()} (Segmentation)')
                else:
                    # Use grayscale for MRI modalities
                    im = ax.imshow(slice_data.T, cmap='gray', origin='lower')
                    ax.set_title(f'{modality.upper()}')
                
                ax.axis('off')
                plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Hide the last subplot if not used
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save:
            output_file = os.path.join(self.output_dir, "01_sample_overview.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Sample overview saved to: {output_file}")
            plt.close()
        else:
            plt.show()
    
    def plot_with_bounding_box(self, save=False):
        """Plot slices with bounding box overlay"""
        if 'seg' not in self.data:
            print("Segmentation mask not available for bounding box calculation.")
            return
        
        # Calculate bounding box
        bbox = self.get_bounding_box(self.data['seg'])
        if bbox is None:
            print("No tumor regions found in segmentation mask.")
            return
        
        min_x, max_x, min_y, max_y, min_z, max_z = bbox
        
        print(f"\nBounding Box Coordinates:")
        print(f"  X: {min_x} - {max_x} (size: {max_x - min_x})")
        print(f"  Y: {min_y} - {max_y} (size: {max_y - min_y})")
        print(f"  Z: {min_z} - {max_z} (size: {max_z - min_z})")
        
        # Plot middle slice with bounding box
        middle_slice = (min_z + max_z) // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Bounding Box Visualization (Slice {middle_slice})', fontsize=16)
        
        modalities_to_plot = ['t1c', 't2w', 'seg']
        
        for i, modality in enumerate(modalities_to_plot):
            if modality in self.data:
                ax = axes[i]
                slice_data = self.data[modality][:, :, middle_slice]
                
                if modality == 'seg':
                    im = ax.imshow(slice_data.T, cmap='tab10', vmin=0, vmax=4, origin='lower')
                else:
                    im = ax.imshow(slice_data.T, cmap='gray', origin='lower')
                
                # Draw bounding box
                rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                   linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                ax.set_title(f'{modality.upper()} with Bounding Box')
                ax.axis('off')
        
        plt.tight_layout()
        
        if save:
            output_file = os.path.join(self.output_dir, "02_bounding_box.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Bounding box visualization saved to: {output_file}")
            plt.close()
        else:
            plt.show()
    
    def plot_segmentation_3d_view(self, save=False):
        """Plot segmentation from different views (axial, sagittal, coronal)"""
        if 'seg' not in self.data:
            print("Segmentation mask not available.")
            return
        
        seg_data = self.data['seg']
        
        # Get middle slices for each view
        axial_slice = seg_data.shape[2] // 2
        sagittal_slice = seg_data.shape[0] // 2
        coronal_slice = seg_data.shape[1] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Segmentation: Multi-View', fontsize=16)
        
        # Axial view (xy plane)
        axes[0].imshow(seg_data[:, :, axial_slice].T, cmap='tab10', vmin=0, vmax=4, origin='lower')
        axes[0].set_title(f'Axial (Z={axial_slice})')
        axes[0].axis('off')
        
        # Sagittal view (yz plane)
        axes[1].imshow(seg_data[sagittal_slice, :, :].T, cmap='tab10', vmin=0, vmax=4, origin='lower')
        axes[1].set_title(f'Sagittal (X={sagittal_slice})')
        axes[1].axis('off')
        
        # Coronal view (xz plane)
        axes[2].imshow(seg_data[:, coronal_slice, :].T, cmap='tab10', vmin=0, vmax=4, origin='lower')
        axes[2].set_title(f'Coronal (Y={coronal_slice})')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save:
            output_file = os.path.join(self.output_dir, "03_multiview_segmentation.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Multi-view segmentation saved to: {output_file}")
            plt.close()
        else:
            plt.show()
    
    def plot_tumor_progression_slices(self, num_slices=9, save=False):
        """Plot tumor progression across multiple slices"""
        if 'seg' not in self.data or 't1c' not in self.data:
            print("Required data not available.")
            return
        
        seg_data = self.data['seg']
        t1c_data = self.data['t1c']
        
        # Find slices with tumor
        tumor_slices = []
        for z in range(seg_data.shape[2]):
            if np.any(seg_data[:, :, z] > 0):
                tumor_slices.append(z)
        
        if not tumor_slices:
            print("No tumor found in segmentation.")
            return
        
        # Select evenly spaced slices
        if len(tumor_slices) >= num_slices:
            selected_slices = [tumor_slices[i * len(tumor_slices) // num_slices] 
                             for i in range(num_slices)]
        else:
            selected_slices = tumor_slices
        
        rows = int(np.ceil(len(selected_slices) / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 5))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Tumor Progression Across Slices (T1C + Segmentation Overlay)', fontsize=16)
        
        for i, slice_idx in enumerate(selected_slices):
            row = i // 3
            col = i % 3
            
            if row < rows and col < 3:
                ax = axes[row, col]
                
                # Plot T1C as background
                t1c_slice = t1c_data[:, :, slice_idx]
                ax.imshow(t1c_slice.T, cmap='gray', origin='lower', alpha=0.8)
                
                # Overlay segmentation
                seg_slice = seg_data[:, :, slice_idx]
                masked_seg = np.ma.masked_where(seg_slice == 0, seg_slice)
                ax.imshow(masked_seg.T, cmap='tab10', vmin=0, vmax=4, origin='lower', alpha=0.7)
                
                ax.set_title(f'Slice {slice_idx}')
                ax.axis('off')
        
        # Hide unused subplots
        for i in range(len(selected_slices), rows * 3):
            row = i // 3
            col = i % 3
            if row < rows and col < 3:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save:
            output_file = os.path.join(self.output_dir, "04_tumor_progression_slices.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Tumor progression visualization saved to: {output_file}")
            plt.close()
        else:
            plt.show()

    def save_all_visualizations(self):
        """Save all visualizations and characteristics to files"""
        print("\n" + "="*80)
        print("SAVING VISUALIZATIONS AND CHARACTERISTICS")
        print("="*80)
        
        # Save characteristics to text file
        self.save_characteristics_to_file()
        
        # Save all visualizations
        print("\nSaving visualizations...")
        self.plot_sample_overview(save=True)
        self.plot_with_bounding_box(save=True)
        self.plot_segmentation_3d_view(save=True)
        self.plot_tumor_progression_slices(save=True)
        
        print(f"\n✓ All files saved in: {self.output_dir}")
        print("Files created:")
        print("  - data_characteristics.txt")
        print("  - 01_sample_overview.png")
        print("  - 02_bounding_box.png") 
        print("  - 03_multiview_segmentation.png")
        print("  - 04_tumor_progression_slices.png")


def main():
    """Main function to demonstrate the BraTS data loader"""
    
    # Dataset configuration
    DATA_ROOT = r"D:\BraTS2024-BraTS-GLI-TrainingData\training_data1_v2"
    SAMPLE_NAME = "BraTS-GLI-00005-100"
    
    print("BraTS2021 Data Loader and Visualizer")
    print("="*80)
    
    # Initialize loader
    loader = BraTSDataLoader(DATA_ROOT, SAMPLE_NAME)
    
    # Load data
    data = loader.load_data()
    
    if not data:
        print("Failed to load data. Please check file paths.")
        return
    
    # Print characteristics
    loader.print_characteristics()
    
    # Save all visualizations and characteristics to files
    loader.save_all_visualizations()
    
    # Optional: Also display visualizations interactively
    print("\n" + "="*80)
    print("DISPLAYING INTERACTIVE VISUALIZATIONS")
    print("="*80)
    
    print("\n1. Sample Overview (All Modalities)")
    loader.plot_sample_overview()
    
    print("\n2. Bounding Box Visualization")
    loader.plot_with_bounding_box()
    
    print("\n3. Multi-View Segmentation")
    loader.plot_segmentation_3d_view()
    
    print("\n4. Tumor Progression Across Slices")
    loader.plot_tumor_progression_slices()
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main() 