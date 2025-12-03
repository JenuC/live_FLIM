# Requirements Document

## Introduction

This document specifies the requirements for extracting the FLIM (Fluorescence Lifetime Imaging Microscopy) data processing logic from the napari-live-flim plugin into a standalone Python library. The library will provide core FLIM analysis capabilities including phasor analysis, curve fitting, and data streaming without dependencies on napari or Qt. The library will be designed for use in Jupyter notebooks and other Python environments, enabling researchers to perform FLIM analysis programmatically.

## Glossary

- **FLIM**: Fluorescence Lifetime Imaging Microscopy - a technique that measures fluorescence decay times
- **Phasor**: A representation of fluorescence lifetime data in the frequency domain using phasor coordinates (g, s)
- **Photon Count**: A 3D array representing the histogram of photon arrival times at each pixel
- **Lifetime Image**: A computed RGB image where color represents fluorescence lifetime and intensity represents photon count
- **RLD**: Rapid Lifetime Determination - a fast curve fitting algorithm using triple integral method
- **LMA**: Levenberg-Marquardt Algorithm - an iterative curve fitting method for more accurate lifetime estimation
- **TCSPC**: Time-Correlated Single Photon Counting - the measurement technique that produces FLIM data
- **Library**: The standalone FLIM processing library being developed
- **Stream Receiver**: A component that receives FLIM data via UDP from acquisition hardware
- **Fitting Engine**: A component that performs curve fitting on fluorescence decay curves

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to process FLIM data streams in real-time, so that I can analyze fluorescence lifetime measurements as they are acquired.

#### Acceptance Criteria

1. WHEN the Library receives a new series message via UDP THEN the Library SHALL initialize a data structure with the specified shape and data type
2. WHEN the Library receives element data via UDP THEN the Library SHALL store the photon count array in memory
3. WHEN the Library receives an end series message THEN the Library SHALL finalize the current series and prepare for the next series
4. WHEN multiple data elements arrive in sequence THEN the Library SHALL process them in the correct order based on sequence numbers
5. WHERE the user specifies a UDP port number, the Library SHALL listen for incoming FLIM data on that port

### Requirement 2

**User Story:** As a researcher, I want to compute phasor coordinates from photon count data, so that I can visualize and analyze fluorescence lifetime in the phasor space.

#### Acceptance Criteria

1. WHEN the Library computes phasor coordinates from photon count data THEN the Library SHALL calculate the g and s coordinates using the Fourier transform method
2. WHEN the Library applies fit range parameters THEN the Library SHALL use only the photon counts within the specified fit_start and fit_end indices
3. WHEN the Library computes phasor coordinates THEN the Library SHALL return an array with shape (height, width, 2) containing g and s values
4. WHEN photon count data contains NaN values THEN the Library SHALL propagate NaN to the corresponding phasor coordinates
5. WHERE the user specifies a laser period, the Library SHALL use that period for phasor calculation

### Requirement 3

**User Story:** As a researcher, I want to perform curve fitting on fluorescence decay data, so that I can extract quantitative lifetime parameters.

#### Acceptance Criteria

1. WHEN the Library performs RLD fitting THEN the Library SHALL compute lifetime using the triple integral method
2. WHEN the Library performs LMA fitting THEN the Library SHALL use the RLD result as the initial parameter estimate
3. WHEN the Library computes curve fits THEN the Library SHALL return tau (lifetime), chi-squared, and fitted curve values
4. WHEN the user specifies fit_start and fit_end parameters THEN the Library SHALL apply fitting only within that range
5. WHERE the fit range exceeds the data size, the Library SHALL adjust the range to the available data

### Requirement 4

**User Story:** As a researcher, I want to generate lifetime images with color-coded lifetimes, so that I can visualize spatial variations in fluorescence lifetime.

#### Acceptance Criteria

1. WHEN the Library generates a lifetime image THEN the Library SHALL compute RLD fitting for each pixel
2. WHEN the Library applies display filters THEN the Library SHALL exclude pixels where chi-squared exceeds max_chisq
3. WHEN the Library applies display filters THEN the Library SHALL exclude pixels where tau is outside the range [min_tau, max_tau]
4. WHEN the Library generates RGB output THEN the Library SHALL encode lifetime as color using the specified colormap
5. WHEN the Library generates RGB output THEN the Library SHALL modulate color intensity by the total photon count

### Requirement 5

**User Story:** As a researcher, I want to select regions of interest in lifetime images or phasor plots, so that I can analyze specific populations of fluorophores.

#### Acceptance Criteria

1. WHEN the Library applies a spatial mask to photon count data THEN the Library SHALL extract photon counts only from masked pixels
2. WHEN the Library computes selection statistics THEN the Library SHALL average the photon counts across all selected pixels
3. WHEN the Library applies a phasor space selection THEN the Library SHALL identify pixels whose phasor coordinates fall within the selection boundary
4. WHEN the Library computes decay curves for a selection THEN the Library SHALL perform both RLD and LMA fitting on the averaged histogram
5. WHERE a selection contains no valid pixels, the Library SHALL return NaN values for all computed parameters

### Requirement 6

**User Story:** As a researcher, I want to work with FLIM data in Jupyter notebooks, so that I can integrate FLIM analysis into my computational workflows.

#### Acceptance Criteria

1. WHEN the Library is imported in a Jupyter notebook THEN the Library SHALL load without requiring napari or Qt dependencies
2. WHEN the Library provides data access methods THEN the Library SHALL return standard NumPy arrays
3. WHEN the Library provides visualization helpers THEN the Library SHALL return data compatible with matplotlib
4. WHEN the Library processes data THEN the Library SHALL provide progress indicators suitable for notebook environments
5. WHERE the user requests computed results, the Library SHALL provide synchronous access to completed computations

### Requirement 7

**User Story:** As a researcher, I want to configure FLIM processing parameters, so that I can optimize analysis for different experimental conditions.

#### Acceptance Criteria

1. WHEN the Library accepts FLIM parameters THEN the Library SHALL validate that period is a positive number
2. WHEN the Library accepts FLIM parameters THEN the Library SHALL validate that fit_start is less than fit_end
3. WHEN the Library accepts display settings THEN the Library SHALL validate that min_tau is less than max_tau
4. WHEN the Library updates parameters THEN the Library SHALL invalidate cached computations that depend on those parameters
5. WHERE parameters are invalid, the Library SHALL raise a descriptive exception

### Requirement 8

**User Story:** As a researcher, I want to compute phasor-based spatial queries efficiently, so that I can perform interactive selection in phasor space.

#### Acceptance Criteria

1. WHEN the Library builds a phasor spatial index THEN the Library SHALL construct a KD-tree from phasor coordinates
2. WHEN the Library queries the phasor spatial index THEN the Library SHALL return pixel indices within the specified distance threshold
3. WHEN the Library handles NaN phasor values THEN the Library SHALL replace them with infinity before building the spatial index
4. WHEN the Library performs phasor queries THEN the Library SHALL use the infinity norm for distance calculations
5. WHERE the query region contains no points, the Library SHALL return an empty array

### Requirement 9

**User Story:** As a researcher, I want to process snapshot and delta snapshot modes, so that I can analyze both cumulative and incremental FLIM data.

#### Acceptance Criteria

1. WHEN the Library operates in snapshot mode THEN the Library SHALL store each received frame independently
2. WHEN the Library operates in delta snapshot mode THEN the Library SHALL compute the difference between consecutive frames
3. WHEN the Library computes a delta snapshot THEN the Library SHALL subtract the previous frame from the current frame
4. WHEN the Library accesses the first frame in delta mode THEN the Library SHALL return the frame without subtraction
5. WHERE no previous frame exists, the Library SHALL return the current frame unchanged

### Requirement 10

**User Story:** As a developer, I want the Library to use asynchronous computation, so that data processing does not block the main thread.

#### Acceptance Criteria

1. WHEN the Library submits computation tasks THEN the Library SHALL use a thread pool executor
2. WHEN the Library returns computation results THEN the Library SHALL provide Future objects for asynchronous access
3. WHEN the Library invalidates tasks THEN the Library SHALL cancel pending futures
4. WHEN the Library checks task completion THEN the Library SHALL provide methods to query future status without blocking
5. WHERE a computation is cancelled, the Library SHALL clean up resources and allow task restart
