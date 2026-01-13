# Mitochondrial Cristae Density Analysis

An ImageJ/Fiji macro for quantifying cristae density in electron microscopy images using the linear intercept method combined with machine learning segmentation.

## Overview

This tool measures **cristae density** (number of cristae per unit area) by:
1. Using a trained Weka classifier to generate probability maps of cristae
2. Allowing users to draw lines across mitochondria and automatically counting cristae crossings
3. Computing density as: `cristae count / mitochondria area`

## Requirements

- **Fiji** (ImageJ with batteries included): https://fiji.sc/
- **Trainable Weka Segmentation** plugin (included with Fiji)
- A trained Weka classifier (`.model` file) for cristae detection
- Minimum 8GB RAM recommended (16GB+ for large datasets)

## Installation

1. Download `Cristae_Linear_Intercept.ijm`
2. In Fiji: `Plugins → Macros → Install...`
3. Select the downloaded `.ijm` file

Or place the file in your Fiji `macros` folder for permanent installation.

## Usage

### Preparation

1. Open your EM image in Fiji
2. Use ROI Manager to outline individual mitochondria (freehand or polygon tool)
3. Ensure you have a trained Weka classifier for cristae segmentation

### Phase 1: Batch Preparation (F1)

Press **F1** to run batch preprocessing:

1. Select your Weka classifier (`.model` file)
2. Choose an output folder
3. Set padding (pixels around each ROI)

The macro will:
- Crop each ROI with padding
- Apply the Weka classifier to generate probability maps
- Save all intermediate files for Phase 2

**Progress is auto-saved** - if Fiji crashes or you need to stop, simply run Phase 1 again and it will skip already-processed ROIs.

### Phase 2: Interactive Analysis (F2)

Press **F2** to analyze cristae:

1. Select the output folder from Phase 1
2. Adjust detection parameters if needed

For each mitochondrion:
1. **Draw a line** across the cristae (perpendicular to cristae orientation)
2. Yellow dots show automatically detected crossings
3. **Adjust the count** if needed
4. Choose an action:
   - **ACCEPT**: Save count and move to next
   - **ADD LINE**: Draw additional lines (counts accumulate)
   - **REDRAW**: Clear markers and start over
   - **SKIP**: Mark as invalid/skip
   - **GO BACK**: Return to previous ROI
   - **QUIT**: Save progress and exit

**Progress auto-saves** after each ROI - you can quit and resume anytime.

### Export Results (E)

Press **E** to export results to CSV with columns:
- `Mito_Label`: ROI name
- `ROI_Area_px2`: Mitochondria area in pixels²
- `Cristae_Count`: Number of cristae intercepts
- `Line_Length_px`: Total line length drawn
- `Density_per_1000px2`: Cristae per 1000 pixels²

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Padding | 20 px | Extra pixels around each ROI crop |
| Threshold | 128 (8-bit) / 0.5 (32-bit) | Probability threshold for cristae detection |
| Min Peak Width | 2 px | Minimum width to count as a crista |
| Min Peak Distance | 3 px | Minimum spacing between cristae |

## Output Structure

```
OutputFolder/
├── raw_crops/           # Original cropped images
│   └── Mito_001_raw.tif
├── probability_maps/    # Weka probability outputs
│   └── Mito_001_prob.tif
├── metadata/            # ROI position data
│   └── Mito_001_meta.txt
├── ROI_set.zip          # Backup of ROI Manager
└── results_autosave.csv # Analysis results
```

## Troubleshooting

### "Could not apply Classifier" error
This usually means the classifier didn't fully load before processing started. The macro includes retry logic and will automatically attempt again. If it persists:
- Increase Fiji's memory: `Edit → Options → Memory & Threads` (set to 10-12GB)
- Restart Fiji and try again

### Memory issues with large datasets
The macro includes automatic garbage collection every 50 ROIs. For very large datasets (500+ ROIs):
- Process in batches of 200-300
- Restart Fiji between batches
- Progress is saved, so you won't lose work

### Weka window doesn't open
- Ensure Trainable Weka Segmentation is installed
- Try: `Help → Update...` to update Fiji
- Restart Fiji and try again

## Training a Classifier

If you don't have a classifier yet:

1. Open a representative EM image
2. `Plugins → Segmentation → Trainable Weka Segmentation`
3. Draw examples:
   - Class 1 (red): Cristae membranes
   - Class 2 (green): Background/matrix
4. Click "Train classifier"
5. Iterate until segmentation looks good
6. `Save classifier` to create your `.model` file

## Citation

If you use this tool in your research, please cite appropriately.

## License

MIT License - feel free to use and modify.

## Contributing

Issues and pull requests welcome!
