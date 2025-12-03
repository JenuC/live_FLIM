# Implementation Plan

- [ ] 1. Set up project structure and core data models




  - Create package directory structure (flim_processing/)
  - Set up pyproject.toml with dependencies (numpy, scipy, flimlib, matplotlib)
  - Implement core dataclasses: FlimParams, DisplaySettings, ProcessingSettings, SeriesMetadata, ElementData
  - Add validation logic in __post_init__ methods
  - _Requirements: 7.1, 7.2, 7.3, 7.5_

- [ ]* 1.1 Write property tests for parameter validation
  - **Property 22: Period validation**
  - **Validates: Requirements 7.1**

- [ ]* 1.2 Write property tests for fit range validation
  - **Property 23: Fit range validation**
  - **Validates: Requirements 7.2**

- [ ]* 1.3 Write property tests for tau range validation
  - **Property 24: Tau range validation**
  - **Validates: Requirements 7.3**

- [ ]* 1.4 Write property tests for descriptive exceptions
  - **Property 26: Descriptive exceptions**
  - **Validates: Requirements 7.5**

- [ ] 2. Implement UDP stream receiver
  - Create StreamReceiver class with socket initialization
  - Implement message parsing (_parse_message, _parse_message_new_series, etc.)
  - Implement memory-mapped file reading (_map_array)
  - Create generator method start_receiving() that yields SeriesMetadata, ElementData, EndSeries events
  - Add stop_receiving() method with socket cleanup
  - _Requirements: 1.1, 1.2, 1.3, 1.5_

- [ ]* 2.1 Write unit tests for UDP receiver
  - Test socket binding to correct port
  - Test message parsing for all message types
  - Test memory-mapped file reading
  - _Requirements: 1.1, 1.3, 1.5_

- [ ] 3. Implement data manager
  - Create DataManager class with snapshot storage
  - Implement add_element() method with sequence number tracking
  - Implement snapshot() method for creating frame copies
  - Implement get_photon_count() with delta mode support
  - Add get_frame_count() method
  - _Requirements: 1.2, 1.4, 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ]* 3.1 Write property test for element storage order
  - **Property 1: Element data storage preserves order**
  - **Validates: Requirements 1.2, 1.4**

- [ ]* 3.2 Write property test for snapshot independence
  - **Property 30: Snapshot mode independence**
  - **Validates: Requirements 9.1**

- [ ]* 3.3 Write property test for delta computation
  - **Property 31: Delta computation correctness**
  - **Validates: Requirements 9.2, 9.3**

- [ ]* 3.4 Write unit tests for edge cases
  - Test first frame in delta mode (no subtraction)
  - Test missing previous frame handling
  - _Requirements: 9.4, 9.5_

- [ ] 4. Implement phasor computation
  - Create PhasorComputer class
  - Implement compute_phasor() using flimlib.GCI_Phasor
  - Apply fit range parameters (fit_start, fit_end)
  - Handle NaN propagation
  - Return array with shape (height, width, 2)
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ]* 4.1 Write property test for phasor shape invariant
  - **Property 2: Phasor output shape invariant**
  - **Validates: Requirements 2.3**

- [ ]* 4.2 Write property test for NaN propagation
  - **Property 3: Phasor NaN propagation**
  - **Validates: Requirements 2.4**

- [ ]* 4.3 Write property test for fit range application
  - **Property 4: Fit range application**
  - **Validates: Requirements 2.2, 3.4**

- [ ]* 4.4 Write property test for period parameter effect
  - **Property 5: Period parameter affects phasor calculation**
  - **Validates: Requirements 2.5**

- [ ]* 4.5 Write property test for Fourier transform correctness
  - **Property 6: Phasor Fourier transform correctness**
  - **Validates: Requirements 2.1**

- [ ] 5. Implement KDTree for phasor queries
  - Implement build_kdtree() method in PhasorComputer
  - Scale phasor coordinates before building tree
  - Replace NaN values with infinity
  - Add query methods using infinity norm
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

- [ ]* 5.1 Write property test for KDTree query correctness
  - **Property 27: KDTree query correctness**
  - **Validates: Requirements 8.2**

- [ ]* 5.2 Write property test for NaN to infinity conversion
  - **Property 28: NaN to infinity conversion**
  - **Validates: Requirements 8.3**

- [ ]* 5.3 Write property test for infinity norm distance
  - **Property 29: Infinity norm distance**
  - **Validates: Requirements 8.4**

- [ ]* 5.4 Write unit test for empty query results
  - Test query region with no points returns empty array
  - _Requirements: 8.5_

- [ ] 6. Implement fitting engine
  - Create FittingEngine class
  - Implement compute_rld() using flimlib.GCI_triple_integral_fitting_engine
  - Implement compute_lma() using flimlib.GCI_marquardt_fitting_engine with RLD initialization
  - Apply fit range parameters
  - Handle fit range exceeding data size
  - Return RLDResult and LMAResult dataclasses
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ]* 6.1 Write property test for RLD triple integral
  - **Property 9: RLD triple integral implementation**
  - **Validates: Requirements 3.1**

- [ ]* 6.2 Write property test for LMA initialization
  - **Property 7: LMA uses RLD initialization**
  - **Validates: Requirements 3.2**

- [ ]* 6.3 Write property test for fitting result completeness
  - **Property 8: Fitting result completeness**
  - **Validates: Requirements 3.3**

- [ ]* 6.4 Write unit test for fit range exceeding data
  - Test that fit range is adjusted when it exceeds data size
  - _Requirements: 3.5_

- [ ] 7. Implement lifetime image generation
  - Implement compute_lifetime_image() in FittingEngine
  - Compute RLD fitting for each pixel
  - Apply display filters (max_chisq, min_tau, max_tau)
  - Apply colormap to tau values
  - Modulate RGB by intensity
  - Return RGBA array
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ]* 7.1 Write property test for RLD per pixel
  - **Property 14: RLD applied per pixel**
  - **Validates: Requirements 4.1**

- [ ]* 7.2 Write property test for chi-squared filtering
  - **Property 10: Display filter excludes high chi-squared**
  - **Validates: Requirements 4.2**

- [ ]* 7.3 Write property test for tau range filtering
  - **Property 11: Display filter excludes out-of-range tau**
  - **Validates: Requirements 4.3**

- [ ]* 7.4 Write property test for colormap application
  - **Property 12: Colormap application**
  - **Validates: Requirements 4.4**

- [ ]* 7.5 Write property test for intensity modulation
  - **Property 13: Intensity modulation**
  - **Validates: Requirements 4.5**

- [ ] 8. Implement asynchronous computation pipeline
  - Create ComputeTask class with Future fields
  - Implement ProcessingPipeline class with ThreadPoolExecutor
  - Implement submit_computation() to create and start tasks
  - Implement task invalidation and cancellation
  - Add is_done(), cancel(), invalidate() methods to ComputeTask
  - Implement update_settings() with cache invalidation
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 7.4_

- [ ]* 8.1 Write property test for Future return type
  - **Property 32: Future return type**
  - **Validates: Requirements 10.2**

- [ ]* 8.2 Write property test for task cancellation
  - **Property 33: Task cancellation**
  - **Validates: Requirements 10.3**

- [ ]* 8.3 Write property test for non-blocking status check
  - **Property 34: Non-blocking status check**
  - **Validates: Requirements 10.4**

- [ ]* 8.4 Write property test for task restart
  - **Property 35: Task restart after cancellation**
  - **Validates: Requirements 10.5**

- [ ]* 8.5 Write property test for cache invalidation
  - **Property 25: Parameter update invalidates cache**
  - **Validates: Requirements 7.4**

- [ ] 9. Implement selection analyzer
  - Create SelectionAnalyzer class
  - Implement analyze_spatial_selection() with mask application
  - Implement analyze_phasor_selection() with KDTree queries
  - Compute averaged histograms across selected pixels
  - Perform RLD and LMA fitting on averaged data
  - Handle empty selections (return NaN)
  - Return SelectionResult dataclass
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ]* 9.1 Write property test for spatial mask extraction
  - **Property 15: Spatial mask extraction**
  - **Validates: Requirements 5.1**

- [ ]* 9.2 Write property test for selection averaging
  - **Property 16: Selection averaging correctness**
  - **Validates: Requirements 5.2**

- [ ]* 9.3 Write property test for phasor selection
  - **Property 17: Phasor space selection correctness**
  - **Validates: Requirements 5.3**

- [ ]* 9.4 Write property test for both fits performed
  - **Property 18: Selection performs both fits**
  - **Validates: Requirements 5.4**

- [ ]* 9.5 Write unit test for empty selection
  - Test that empty selection returns NaN values
  - _Requirements: 5.5_

- [ ] 10. Implement high-level FlimSession interface
  - Create FlimSession class
  - Integrate StreamReceiver, DataManager, ProcessingPipeline, SelectionAnalyzer
  - Implement start_streaming() and stop_streaming()
  - Implement load_data() for array input
  - Implement get_lifetime_image() and get_phasor() with synchronous access
  - Implement create_spatial_selection() and create_phasor_selection()
  - Implement snapshot() method
  - Add settings property with getter/setter
  - _Requirements: 6.1, 6.2, 6.3, 6.5_

- [ ]* 10.1 Write property test for NumPy array returns
  - **Property 19: Data access returns NumPy arrays**
  - **Validates: Requirements 6.2**

- [ ]* 10.2 Write property test for matplotlib compatibility
  - **Property 20: Matplotlib compatibility**
  - **Validates: Requirements 6.3**

- [ ]* 10.3 Write property test for synchronous access
  - **Property 21: Synchronous result access**
  - **Validates: Requirements 6.5**

- [ ]* 10.4 Write unit test for import without napari/Qt
  - Test that importing library doesn't trigger napari or Qt imports
  - _Requirements: 6.1_

- [ ] 11. Add colormap support
  - Load matplotlib colormaps
  - Load custom colormaps from CSV files (BH_compat, etc.)
  - Create colormap registry
  - Implement colormap application in lifetime image generation
  - _Requirements: 4.4_

- [ ]* 11.1 Write unit tests for colormap loading
  - Test matplotlib colormap access
  - Test custom colormap loading from CSV
  - _Requirements: 4.4_

- [ ] 12. Create example notebooks
  - Create basic usage example notebook
  - Create streaming example notebook
  - Create selection analysis example notebook
  - Add visualization examples with matplotlib
  - _Requirements: 6.1, 6.3_

- [ ] 13. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Add documentation and package metadata
  - Write README.md with installation and usage instructions
  - Add docstrings to all public classes and methods
  - Create API reference documentation
  - Set up package metadata in pyproject.toml
  - _Requirements: All_

- [ ] 15. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.
