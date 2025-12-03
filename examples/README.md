# FLIM Processing Library - Example Notebooks

This directory contains Jupyter notebooks demonstrating the usage of the FLIM Processing Library.

## Overview

The FLIM Processing Library is a standalone Python package for Fluorescence Lifetime Imaging Microscopy (FLIM) data analysis. These examples show how to use the library for various FLIM analysis tasks without requiring napari or Qt dependencies.

## Notebooks

### 1. Basic Usage (`01_basic_usage.ipynb`)

**Topics covered:**
- Generating synthetic FLIM data
- Computing phasor coordinates
- Performing curve fitting (RLD and LMA)
- Visualizing lifetime maps and phasor plots
- Using the DataManager for data organization

**Best for:** Getting started with the library and understanding core concepts.

### 2. Streaming Example (`02_streaming_example.ipynb`)

**Topics covered:**
- Setting up UDP stream reception
- Processing real-time FLIM data
- Handling SeriesMetadata, ElementData, and EndSeries events
- Creating snapshots during streaming
- Delta snapshot mode for dynamic samples

**Best for:** Working with live FLIM data from TCSPC hardware.

### 3. Selection Analysis (`03_selection_analysis.ipynb`)

**Topics covered:**
- Creating spatial selections (rectangular, circular, custom shapes)
- Performing phasor space selections using KDTree
- Analyzing regions of interest
- Comparing spatial vs phasor-based selection methods
- Extracting quantitative lifetime parameters from selections

**Best for:** Region-of-interest analysis and identifying fluorophore populations.

## Requirements

To run these notebooks, you need:

```bash
pip install flim-processing numpy matplotlib jupyter
```

Optional dependencies:
- `scipy` - For KDTree-based phasor queries (included in main dependencies)
- `flimlib` - For curve fitting algorithms (included in main dependencies)

## Running the Notebooks

1. Install the FLIM Processing Library and dependencies
2. Navigate to the examples directory
3. Start Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open any notebook and run the cells sequentially

## Data Requirements

### Basic Usage and Selection Analysis
These notebooks generate synthetic FLIM data, so no external data files are required.

### Streaming Example
The streaming example requires a data sender to simulate or provide real FLIM data via UDP. You can:
- Use the `SeriesSender` class from the napari-live-flim package for testing
- Connect to actual TCSPC hardware
- Modify the notebook to load data from files instead

## Key Concepts

### FLIM Parameters
- **Period**: Laser repetition period in nanoseconds
- **Fit Range**: Time bins used for fitting (fit_start to fit_end)
- **Time Bins**: Number of time channels in the TCSPC histogram

### Phasor Analysis
- **g coordinate**: Real part of the Fourier transform (cosine component)
- **s coordinate**: Imaginary part of the Fourier transform (sine component)
- **Universal Circle**: Theoretical boundary for single-exponential decays

### Curve Fitting
- **RLD (Rapid Lifetime Determination)**: Fast triple integral method
- **LMA (Levenberg-Marquardt)**: Iterative refinement using RLD initialization

## Tips for Best Results

1. **Fit Range Selection**: Exclude initial bins affected by the instrument response function (IRF)
2. **Chi-Squared Filtering**: Use chi-squared values to identify poor fits
3. **Tau Range Filtering**: Set reasonable lifetime bounds based on your fluorophores
4. **Phasor Queries**: Adjust the radius parameter based on your data's noise level
5. **Selection Size**: Larger selections provide better statistics but less spatial resolution

## Troubleshooting

### Import Errors
If you get import errors, ensure the library is installed:
```bash
pip install -e /path/to/flim-processing
```

### UDP Connection Issues
For streaming examples:
- Check that the port is not already in use
- Verify firewall settings allow UDP traffic
- Ensure the sender and receiver use the same port number

### Memory Issues
For large datasets:
- Process data in chunks
- Use delta snapshot mode to reduce memory usage
- Consider downsampling spatial dimensions

## Further Reading

- [FLIM Processing Library Documentation](../README.md)
- [Design Document](.kiro/specs/flim-processing-library/design.md)
- [Requirements Document](.kiro/specs/flim-processing-library/requirements.md)

## Contributing

If you create additional example notebooks, please:
1. Follow the existing structure and style
2. Include clear explanations and visualizations
3. Test with both synthetic and real data
4. Add appropriate documentation

## License

These examples are provided under the same license as the FLIM Processing Library.
