"""
Unit tests for core dataclasses and their validation logic.
"""

import pytest
import numpy as np
from flim_processing import (
    FlimParams,
    DisplaySettings,
    ProcessingSettings,
    SeriesMetadata,
    ElementData,
    RLDResult,
    LMAResult,
    SelectionResult,
    EndSeries,
)


class TestFlimParams:
    """Tests for FlimParams validation."""
    
    def test_valid_params(self):
        """Test that valid parameters are accepted."""
        params = FlimParams(period=0.04, fit_start=0, fit_end=256)
        assert params.period == 0.04
        assert params.fit_start == 0
        assert params.fit_end == 256
    
    def test_negative_period_raises_error(self):
        """Test that negative period raises ValueError."""
        with pytest.raises(ValueError, match="period must be positive"):
            FlimParams(period=-0.04, fit_start=0, fit_end=256)
    
    def test_zero_period_raises_error(self):
        """Test that zero period raises ValueError."""
        with pytest.raises(ValueError, match="period must be positive"):
            FlimParams(period=0, fit_start=0, fit_end=256)
    
    def test_fit_start_greater_than_end_raises_error(self):
        """Test that fit_start >= fit_end raises ValueError."""
        with pytest.raises(ValueError, match="fit_start.*must be less than fit_end"):
            FlimParams(period=0.04, fit_start=100, fit_end=50)
    
    def test_fit_start_equal_to_end_raises_error(self):
        """Test that fit_start == fit_end raises ValueError."""
        with pytest.raises(ValueError, match="fit_start.*must be less than fit_end"):
            FlimParams(period=0.04, fit_start=100, fit_end=100)


class TestDisplaySettings:
    """Tests for DisplaySettings validation."""
    
    def test_valid_settings(self):
        """Test that valid settings are accepted."""
        settings = DisplaySettings(
            max_chisq=2.0,
            min_tau=0.5,
            max_tau=4.0,
            colormap="viridis"
        )
        assert settings.max_chisq == 2.0
        assert settings.min_tau == 0.5
        assert settings.max_tau == 4.0
        assert settings.colormap == "viridis"
    
    def test_min_tau_greater_than_max_raises_error(self):
        """Test that min_tau > max_tau raises ValueError."""
        with pytest.raises(ValueError, match="min_tau.*must be less than max_tau"):
            DisplaySettings(
                max_chisq=2.0,
                min_tau=4.0,
                max_tau=0.5,
                colormap="viridis"
            )
    
    def test_min_tau_equal_to_max_raises_error(self):
        """Test that min_tau == max_tau raises ValueError."""
        with pytest.raises(ValueError, match="min_tau.*must be less than max_tau"):
            DisplaySettings(
                max_chisq=2.0,
                min_tau=2.0,
                max_tau=2.0,
                colormap="viridis"
            )


class TestProcessingSettings:
    """Tests for ProcessingSettings."""
    
    def test_valid_processing_settings(self):
        """Test that valid processing settings are accepted."""
        flim_params = FlimParams(period=0.04, fit_start=0, fit_end=256)
        display_settings = DisplaySettings(
            max_chisq=2.0,
            min_tau=0.5,
            max_tau=4.0,
            colormap="viridis"
        )
        settings = ProcessingSettings(
            flim_params=flim_params,
            display_settings=display_settings,
            delta_snapshots=False
        )
        assert settings.flim_params == flim_params
        assert settings.display_settings == display_settings
        assert settings.delta_snapshots is False
    
    def test_delta_snapshots_default(self):
        """Test that delta_snapshots defaults to False."""
        flim_params = FlimParams(period=0.04, fit_start=0, fit_end=256)
        display_settings = DisplaySettings(
            max_chisq=2.0,
            min_tau=0.5,
            max_tau=4.0,
            colormap="viridis"
        )
        settings = ProcessingSettings(
            flim_params=flim_params,
            display_settings=display_settings
        )
        assert settings.delta_snapshots is False


class TestSeriesMetadata:
    """Tests for SeriesMetadata."""
    
    def test_valid_metadata(self):
        """Test that valid metadata is accepted."""
        metadata = SeriesMetadata(
            series_no=1,
            port=4444,
            shape=(256, 256, 128),
            dtype=np.float32
        )
        assert metadata.series_no == 1
        assert metadata.port == 4444
        assert metadata.shape == (256, 256, 128)
        assert metadata.dtype == np.float32


class TestElementData:
    """Tests for ElementData."""
    
    def test_valid_element_data(self):
        """Test that valid element data is accepted."""
        frame = np.zeros((256, 256, 128), dtype=np.float32)
        element = ElementData(
            series_no=1,
            seqno=0,
            frame=frame
        )
        assert element.series_no == 1
        assert element.seqno == 0
        assert element.frame.shape == (256, 256, 128)


class TestEndSeries:
    """Tests for EndSeries."""
    
    def test_valid_end_series(self):
        """Test that valid end series marker is accepted."""
        end = EndSeries(series_no=1)
        assert end.series_no == 1


class TestRLDResult:
    """Tests for RLDResult."""
    
    def test_valid_rld_result(self):
        """Test that valid RLD result is accepted."""
        result = RLDResult(
            tau=np.array([2.0]),
            chisq=np.array([1.5]),
            Z=np.array([10.0]),
            A=np.array([100.0]),
            fitted=np.array([1, 2, 3])
        )
        assert result.tau[0] == 2.0
        assert result.chisq[0] == 1.5
        assert result.Z[0] == 10.0
        assert result.A[0] == 100.0
        assert len(result.fitted) == 3


class TestLMAResult:
    """Tests for LMAResult."""
    
    def test_valid_lma_result(self):
        """Test that valid LMA result is accepted."""
        result = LMAResult(
            param=np.array([10.0, 100.0, 2.0]),
            chisq=np.array([1.2]),
            fitted=np.array([1, 2, 3])
        )
        assert result.param[0] == 10.0
        assert result.param[1] == 100.0
        assert result.param[2] == 2.0
        assert result.chisq[0] == 1.2


class TestSelectionResult:
    """Tests for SelectionResult."""
    
    def test_valid_selection_result(self):
        """Test that valid selection result is accepted."""
        rld = RLDResult(
            tau=np.array([2.0]),
            chisq=np.array([1.5]),
            Z=np.array([10.0]),
            A=np.array([100.0])
        )
        lma = LMAResult(
            param=np.array([10.0, 100.0, 2.0]),
            chisq=np.array([1.2])
        )
        result = SelectionResult(
            histogram=np.array([1, 2, 3]),
            points=np.array([[0, 0], [1, 1]]),
            rld=rld,
            lma=lma,
            pixel_count=2
        )
        assert len(result.histogram) == 3
        assert result.points.shape == (2, 2)
        assert result.pixel_count == 2
