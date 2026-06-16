# BraTS2021 Data Loader and Visualizer

This folder contains a comprehensive tool for loading and visualizing BraTS2021 brain tumor segmentation data.

## Files

- `load_brats_sample.py` - Main script for loading and visualizing BraTS data
- `requirements.txt` - Required Python packages
- `README.md` - This file

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Update the file paths in `load_brats_sample.py`:
   - `DATA_ROOT`: Path to your BraTS dataset root directory
   - `SAMPLE_NAME`: Name of the sample to analyze

2. Run the script:
```bash
python load_brats_sample.py
```

## Features

### Data Loading
- Loads all MRI modalities (T1c, T1n, T2w, T2f)
- Loads segmentation masks
- Handles both .nii and .nii.gz formats
- Provides detailed error handling and status reporting

### Data Characteristics
- Prints detailed statistics for each modality
- Shows data shapes, types, and value ranges
- Analyzes segmentation label distributions
- Displays voxel spacing and orientation information

### Visualizations

1. **Sample Overview**: Shows all modalities for a middle slice
2. **Bounding Box Visualization**: Displays tumor bounding boxes with coordinates
3. **Multi-View Segmentation**: Shows segmentation from axial, sagittal, and coronal views
4. **Tumor Progression**: Shows tumor across multiple slices with overlay

## Dataset Structure Expected

```
DATA_ROOT/
└── SAMPLE_NAME/
    ├── SAMPLE_NAME-t1c.nii.gz
    ├── SAMPLE_NAME-t1n.nii.gz
    ├── SAMPLE_NAME-t2w.nii.gz
    ├── SAMPLE_NAME-t2f.nii.gz
    └── SAMPLE_NAME-seg.nii.gz
```

## Customization

You can easily customize the script by:
- Changing the sample name in the `main()` function
- Modifying visualization parameters (slice numbers, colors, etc.)
- Adding new visualization functions
- Adjusting bounding box margins

## Output

The script will:
1. Print loading status for each file
2. Display comprehensive data characteristics
3. Generate multiple visualization plots
4. Show bounding box coordinates
5. Display tumor progression across slices

All visualizations are displayed using matplotlib and can be saved manually from the plot windows. 