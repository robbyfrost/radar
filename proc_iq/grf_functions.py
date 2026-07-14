# --------------------------------------------------
# Name: functions.py
# Author: J.W. Thiesing and Robby M. Frost
# Advanced Radar Research Center
# University of Oklahoma
# Created: 23 Oct 2025
# Purpose: Functions for processing I/Q and 
# clutter filtering. Clutter filter code adapted
# from Vitor Goede
# --------------------------------------------------
import numpy as np
import xarray as xr
import struct
from numpy.polynomial import polynomial as P
from scipy import signal
from tqdm.auto import tqdm
from numpy.typing import NDArray
from datetime import datetime
import pytz
import netCDF4
# ----------------------------------------------------------------------------------------------------
# clutter filtering
# ----------------------------------------------------------------------------------------------------

class GroundClutterFilter:
    """
    Implements a ground clutter filter based on global polynomial regression,
    as described in the provided article "A New Paradigm for Automated Ground
    Clutter Removal: Global Regression Filtering" by Hubbert et al. (2025).

    This filter operates on I (in-phase) and Q (quadrature) time series data.
    It identifies clutter by fitting a polynomial to the time series and
    subtracting this fitted trend. It also includes automated polynomial
    order selection and an optional Gaussian interpolation for the
    zero-velocity gap.

    Python version created by Vitor Goede.
    """

    def __init__(self, wavelength, scan_rate, prt, num_samples):
        """
        Defines important parameters for initializing the clutter filtering.

        Args:
            wavelength (float): Radar wavelength in meters.
            scan_rate (float): antenna rotation speed in degrees/second.
            prt (float): pulse repetition time in 1/s.
            num_samples (float): number of samples per ray.
        """
        self.wavelength = wavelength
        self.scan_rate = scan_rate
        self.prt = prt
        self.num_samples = num_samples
        self.nyquist_velocity = self.wavelength / (4 * self.prt)

        print(f"Filter initialized with:")
        print(f"  Wavelength: {self.wavelength:.4f} m")
        print(f"  Scan Rate: {self.scan_rate} deg/s")
        print(f"  PRT: {self.prt} s")
        print(f"  Number of Samples per gate: {self.num_samples}")
        print(f"  Nyquist Velocity: {self.nyquist_velocity:.2f} m/s")

    def _estimate_clutter_power_from_polyfit(self, i_series, q_series, order=2):
        """
        Estimates the power backscattered by the ground clutter

        Args:
            i_series (np.ndarray): in-phase time series of complex numbers.
            q_series (np.ndarray): quadratyre time series of complex numbers.
            order (int): polynomial fit order.

        Returns:
            np.ndarray: Total clutter power array.
        """
        print(f"  _estimate_clutter_power_from_polyfit: i_series.shape = {i_series.shape}")
        print(f"  _estimate_clutter_power_from_polyfit: q_series.shape = {q_series.shape}")
        t = np.arange(self.num_samples)
        print(f"  _estimate_clutter_power_from_polyfit: t.shape = {t.shape}")
        i_series_T = i_series.T
        q_series_T = q_series.T
        print(f"  _estimate_clutter_power_from_polyfit: i_series_T.shape = {i_series_T.shape}")
        coeffs_i = P.polyfit(t, i_series_T, order)
        print(f"  _estimate_clutter_power_from_polyfit: coeffs_i.shape = {coeffs_i.shape}")

        clutter_i = P.polyval(t, coeffs_i)
        print(f"  _estimate_clutter_power_from_polyfit: clutter_i.shape = {clutter_i.shape}")

        coeffs_q = P.polyfit(t, q_series_T, order)
        clutter_q = P.polyval(t, coeffs_q) # Same for Q component
        print(f"  _estimate_clutter_power_from_polyfit: clutter_q.shape = {clutter_q.shape}")

        clutter_power_i = np.mean(clutter_i**2, axis=-1)
        clutter_power_q = np.mean(clutter_q**2, axis=-1)
        print(f"  _estimate_clutter_power_from_polyfit: clutter_power_i.shape = {clutter_power_i.shape}")

        total_clutter_power = clutter_power_i + clutter_power_q
        print(f"  _estimate_clutter_power_from_polyfit: total_clutter_power.shape = {total_clutter_power.shape}")
        return total_clutter_power

    def _get_polynomial_order(self, cnr_db, b_factor=1.0):
        """
        Determines the required polynomial order for the regression filter
        based on CNR and radar parameters, using the empirical formulas
        from Appendix A of the article.

        Args:
            cnr_db (np.ndarray): Clutter-to-Noise Ratio in dB (1D array or scalar).
            b_factor (float): Multiplicative parameter for clutter spectrum width.

        Returns:
            np.ndarray: The calculated polynomial order (integer array).
        """
        wc = b_factor * (0.03 + 0.017 * self.scan_rate)
        wcn = wc / self.nyquist_velocity

        On = -2.0428 * (wcn**2) + 0.6490 * wcn

        cnr_linear = 10**(cnr_db / 10.0)
        cnr_linear = np.maximum(cnr_linear, 1e-5)

        order = np.ceil(On * (cnr_linear**(2/3.0)) * self.num_samples)

        order = np.maximum(1, np.minimum(order, self.num_samples - 1)).astype(int)
        return order

    def _perform_regression_filter(self, time_series_batch, orders):
        """
        Performs the regression filter

        Args:
            time_series_batch (np.ndarray): Batch I/Q time series.
            orders (np.ndarray): Polynomial order.

        Returns:
           tuple: (filtered_batch, clutter_trend_map)
                    filtered_batch (np.ndarray): Clutter-suppressed I/Q.
                    orders (np.ndarray): Polynomial orders.
        """
        filtered_batch = np.zeros_like(time_series_batch)
        clutter_trend_map = np.zeros_like(time_series_batch)
        unique_orders = np.unique(orders)

        for order in unique_orders:
            indices_for_this_order = np.where(orders == order)[0]
            if len(indices_for_this_order) == 0:
                continue

            selected_series = time_series_batch[indices_for_this_order, :]
            
            t = np.arange(self.num_samples)
            series_T = selected_series.T

            coeffs = P.polyfit(t, series_T, order)

            clutter_trend = P.polyval(t, coeffs)

            print(f"  _perform_regression_filter: selected_series.shape = {selected_series.shape}")
            print(f"  _perform_regression_filter: clutter_trend.shape = {clutter_trend.shape}")

            filtered_series = selected_series - clutter_trend

            filtered_batch[indices_for_this_order, :] = filtered_series
            clutter_trend_map[indices_for_this_order, :] = clutter_trend

        return filtered_batch, clutter_trend_map

    def _gaussian_interpolation_batch(self, spectrum_batch, poly_orders, E_points=3, velocity_threshold_interp=0.2):
        """
        Applies Gaussian interpolation across the zero-velocity gap in the
        Doppler spectrum for a batch of spectra.

        Args:
            spectrum_batch (np.ndarray): Batch of Doppler power spectra (linear units) (num_series, num_samples).
            poly_orders (np.ndarray): Polynomial order used for filtering for each series (1D array).
            E_points (int): Number of points on either side of the gap used for Gaussian fitting.
            velocity_threshold_interp (float): Threshold for estimated velocity relative to Nyquist velocity.

        Returns:
            np.ndarray: Spectra with the zero-velocity gap interpolated (num_series, num_samples).
        """
        num_series, num_samples = spectrum_batch.shape
        interpolated_spectrum_batch = np.copy(spectrum_batch)

        zero_vel_idx = num_samples // 2
        nyquist_velocity_per_bin = (2 * self.nyquist_velocity / num_samples)

        mean_vel_indices = np.argmax(spectrum_batch, axis=-1)
        estimated_velocities = (mean_vel_indices - zero_vel_idx) * nyquist_velocity_per_bin

        apply_interp_mask = (np.abs(estimated_velocities / self.nyquist_velocity) <= velocity_threshold_interp)

        L_values = np.array([self._get_interpolation_width_L(order) for order in poly_orders])

        indices_to_interp = np.where(apply_interp_mask)[0]

        if len(indices_to_interp) == 0:
            return interpolated_spectrum_batch

        for idx in tqdm(indices_to_interp, desc="Batch Gaussian Interpolation"):
            spectrum = spectrum_batch[idx, :]
            L = L_values[idx]

            gap_start_idx = zero_vel_idx - L
            gap_end_idx = zero_vel_idx + L + 1

            interp_start_left = zero_vel_idx - L - E_points
            interp_end_left = zero_vel_idx - L

            interp_start_right = zero_vel_idx + L + 1
            interp_end_right = zero_vel_idx + L + 1 + E_points

            interp_indices_left = np.arange(max(0, interp_start_left), max(0, interp_end_left))
            interp_indices_right = np.arange(min(num_samples, interp_start_right), min(num_samples, interp_end_right))
            interp_indices = np.concatenate((interp_indices_left, interp_indices_right))

            if len(interp_indices) < 2:
                continue

            interp_values = spectrum[interp_indices]
            relative_vel_indices = interp_indices - zero_vel_idx

            PT = np.sum(interp_values)
            if PT == 0:
                continue

            a = np.sum(relative_vel_indices * interp_values) / PT
            s_squared = np.sum(interp_values * (relative_vel_indices - a)**2) / PT
            s = np.sqrt(s_squared)
            s_epsilon = 1e-9
            s = s + s_epsilon if s < s_epsilon else s

            gap_indices = np.arange(gap_start_idx, gap_end_idx)
            relative_gap_indices = gap_indices - zero_vel_idx

            for _ in range(2):
                estimated_gaussian = (PT / (s * np.sqrt(2 * np.pi))) * \
                                     np.exp(-(relative_gap_indices - a)**2 / (2 * s**2))
                interpolated_spectrum_batch[idx, gap_indices] = estimated_gaussian

                combined_indices = np.concatenate((interp_indices, gap_indices))
                combined_values = interpolated_spectrum_batch[idx, combined_indices]
                combined_relative_vel_indices = combined_indices - zero_vel_idx

                PT = np.sum(combined_values)
                if PT == 0:
                    break
                a = np.sum(combined_relative_vel_indices * combined_values) / PT
                s_squared = np.sum(combined_values * (combined_relative_vel_indices - a)**2) / PT
                s = np.sqrt(s_squared)
                s = s + s_epsilon if s < s_epsilon else s
                if s == 0:
                    break
        return interpolated_spectrum_batch

    # def _get_interpolation_width_L(self, poly_order):
    #     """
    #     Determines the interpolation half-width L based on polynomial order.
    #     This is a simplified lookup based on Table B1.
    #     In a real application, this would be derived from the filter's
    #     frequency response.

    #     Args:
    #         poly_order (int): The polynomial order used for filtering.

    #     Returns:
    #         int: The half-width L for interpolation.
    #     """
    #     if poly_order <= 3: return 1
    #     elif poly_order <= 5: return 2
    #     elif poly_order <= 8: return 3
    #     elif poly_order <= 10: return 4
    #     elif poly_order <= 13: return 5
    #     elif poly_order <= 16: return 6
    #     elif poly_order <= 19: return 7
    #     else: return 8

    def _get_interpolation_width_L(self, poly_order):
        """
        Determines the interpolation half-width L based on polynomial order.
        This is a simplified lookup based on Table B1.
        In a real application, this would be derived from the filter's
        frequency response.

        Args:
            poly_order (int): The polynomial order used for filtering.

        Returns:
            int: The half-width L for interpolation.
        """
        if poly_order <= 3: return 1
        elif poly_order <= 5: return 2
        elif poly_order <= 8: return 3
        elif poly_order <= 10: return 4
        elif poly_order <= 13: return 5
        elif poly_order <= 16: return 6
        elif poly_order <= 19: return 7
        elif poly_order <= 22: return 8
        elif poly_order <= 25: return 9
        else: return 10

    def filter_iq_data(self, i_data, q_data, cnr_db_map=None, apply_interpolation=True,
                             interpolation_E_points=3, velocity_threshold_interp=0.2, 
                             b_factor=1.0):
        """
        Main method to filter I/Q time series data for ground clutter.
        This method now handles 3D input data (range, azimuth, sample)
        by flattening the first two dimensions for batch processing.

        Args:
            i_data (np.ndarray): Input in-phase time series of shape (num_ranges, num_azimuths, num_samples).
            q_data (np.ndarray): Input quadrature time series of shape (num_ranges, num_azimuths, num_samples).
            cnr_db_map (np.ndarray, optional): 2D array of CNR in dB of shape (num_ranges, num_azimuths).
                                                If None, CNR will be estimated for each (range, azimuth) cell.
            apply_interpolation (bool): Whether to apply Gaussian interpolation
                                        across the zero-velocity gap. Defaults to True.
            interpolation_E_points (int): Number of points on either side of the
                                          gap used for Gaussian interpolation (E).
                                          Defaults to 3.
            velocity_threshold_interp (float): Threshold for estimated velocity
                                              relative to Nyquist velocity. If
                                              abs(Vest/Nyq) > Vth, interpolation
                                              is not applied. Defaults to 0.2.

        Returns:
            tuple: (filtered_i, filtered_q, poly_order_map, clutter_i_trend_map,
                    clutter_q_trend_map, interpolated_spectrum_map)
                    filtered_i (np.ndarray): Clutter-suppressed I series (num_ranges, num_azimuths, num_samples).
                    filtered_q (np.ndarray): Clutter-suppressed Q series (num_ranges, num_azimuths, num_samples).
                    poly_order_map (np.ndarray): Polynomial order used for filtering for each cell (num_ranges, num_azimuths).
                    clutter_i_trend_map (np.ndarray): Estimated clutter trend for I series (num_ranges, num_azimuths, num_samples).
                    clutter_q_trend_map (np.ndarray): Estimated clutter trend for Q series (num_ranges, num_azimuths, num_samples).
                    interpolated_spectrum_map (np.ndarray): Interpolated Doppler spectrum for each cell
                                                             (num_ranges, num_azimuths, num_samples).
        """
        if i_data.shape != q_data.shape:
            raise ValueError("i_data and q_data must have the same shape.")
        if i_data.ndim != 3 or i_data.shape[2] != self.num_samples:
            raise ValueError(f"Input data must be of shape (range, azimuth, sample) "
                             f"where sample dimension is {self.num_samples}.")

        original_shape = i_data.shape
        num_ranges, num_azimuths, num_samples = original_shape
        num_cells = num_ranges * num_azimuths

        i_data_flat = i_data.reshape(num_cells, num_samples)
        q_data_flat = q_data.reshape(num_cells, num_samples)

        print(f"Processing {num_cells} cells ({num_ranges} range gates, {num_azimuths} azimuthal positions)...")

        filtered_i_flat = np.zeros_like(i_data_flat, dtype=float)
        filtered_q_flat = np.zeros_like(q_data_flat, dtype=float)
        poly_order_flat = np.zeros(num_cells, dtype=int)
        clutter_i_trend_flat = np.zeros_like(i_data_flat, dtype=float)
        clutter_q_trend_flat = np.zeros_like(q_data_flat, dtype=float)
        interpolated_spectrum_flat = np.zeros_like(i_data_flat, dtype=float)

        if cnr_db_map is None:
            print("Estimating CNR for all cells...")
            clutter_power_linear_flat = self._estimate_clutter_power_from_polyfit(i_data_flat, q_data_flat)
            total_raw_power_flat = np.mean(i_data_flat**2 + q_data_flat**2, axis=-1)
            noise_power_linear_flat = np.maximum(1e-10, total_raw_power_flat * 0.1)
            
            cnr_db_flat = np.where(noise_power_linear_flat > 1e-10,
                                   10 * np.log10(clutter_power_linear_flat / noise_power_linear_flat),
                                   -99.0)
        else:
            cnr_db_flat = cnr_db_map.flatten()

        print("Determining polynomial orders...")
        poly_order_flat = self._get_polynomial_order(cnr_db_flat, b_factor=b_factor)
        
        print("Performing regression filtering on I series...")
        filtered_i_flat, clutter_i_trend_flat = self._perform_regression_filter(i_data_flat, poly_order_flat)
        print("Performing regression filtering on Q series...")
        filtered_q_flat, clutter_q_trend_flat = self._perform_regression_filter(q_data_flat, poly_order_flat)

        if apply_interpolation:
            print("Applying Gaussian interpolation (if conditions met)...")
            complex_signal_flat = filtered_i_flat + 1j * filtered_q_flat

            doppler_spectrum_flat = np.abs(np.fft.fftshift(np.fft.fft(complex_signal_flat, axis=-1), axes=-1))**2

            interpolated_spectrum_flat = self._gaussian_interpolation_batch(
                doppler_spectrum_flat,
                poly_order_flat,
                E_points=interpolation_E_points,
                velocity_threshold_interp=velocity_threshold_interp
            )
        else: 
            
            complex_signal_flat = filtered_i_flat + 1j * filtered_q_flat
            interpolated_spectrum_flat = np.abs(np.fft.fftshift(np.fft.fft(complex_signal_flat, axis=-1), axes=-1))**2

        filtered_i = filtered_i_flat.reshape(original_shape)
        filtered_q = filtered_q_flat.reshape(original_shape)
        poly_order_map = poly_order_flat.reshape(original_shape[:-1])
        clutter_i_trend_map = clutter_i_trend_flat.reshape(original_shape)
        clutter_q_trend_map = clutter_q_trend_flat.reshape(original_shape)
        interpolated_spectrum_map = interpolated_spectrum_flat.reshape(original_shape) 
        return filtered_i, filtered_q, poly_order_map, clutter_i_trend_map, clutter_q_trend_map, interpolated_spectrum_map



# ----------------------------------------------------------------------------------------------------
# rfcfile
# ----------------------------------------------------------------------------------------------------

class rkcfile:
    def __init__(self, filename, maxPulse=None, posFilename=None, verbose=True):
        
        #------Class Properties---------------------------------------------------------------------
        self.constants = {
            'RKName': 128,
            'RKFileHeader': 4096,
            'RKMaxMatchedFilterCount': 8,
            'RKFilterAnchorSize': 64,
            'RKMaximumStringLength': 4096,
            'RKMaximumPathLength': 1024,
            'RKMaximumPrefixLength': 8,
            'RKMaximumFolderPathLength': 768,
            'RKMaximumWaveformCount': 22,
            'RKMaximumFilterCount': 8,
            'RKRadarDescOffset': 256,
            'RKRadarDesc': 1072,
            'RKConfigV1': 1441,
            'RKConfig': 1024,
            'RKMaximumCommandLength': 512,
            'RKMaxFilterCount': 8,
            'RKPulseHeaderV1': 256,
            'RKPulseHeader': 384,
            'RKWaveFileGlobalHeader': 512
        }
        
        self.filename = ""
        
        self.header = {
            'preface': [],
            'buildNo': 6,
            'dataType': [],
            'desc': [],
            'config': [],
            'waveform': []
        }
        
        self.pulses = []
        #-------------------------------------------------------------------------------------------
        
        
        #------Set Filename Property----------------------------------------------------------------
        if verbose:
            print(f"Filename: {filename}")
        self.filename = filename
        #-------------------------------------------------------------------------------------------
        
        
        #------Open file for partial reads----------------------------------------------------------
        file = open(self.filename, 'rb')
        #-------------------------------------------------------------------------------------------
        
        
        #------Get preface and build number---------------------------------------------------------
        self.header['preface'] = struct.unpack(f"{self.constants['RKName']}s", 
            file.read(self.constants['RKName']))[0].decode()\
            .replace('\x00', ' ').strip()
        self.header['buildNo'] = struct.unpack('I', file.read(4))[0]
        if verbose:
            print(f"preface = {self.header['preface']} "
                f"  buildNo = {self.header['buildNo']}")
        #-------------------------------------------------------------------------------------------
        
        
        #------If build 5 or higher, get dataType---------------------------------------------------
        if self.header['buildNo'] >= 5:
            self.header['dataType'] = struct.unpack('B', file.read(1))[0]
        #-------------------------------------------------------------------------------------------


        #------Radar Description--------------------------------------------------------------------
        if self.header['buildNo'] >= 2:
            if self.header['buildNo'] >= 6:
                offset = self.constants['RKRadarDescOffset']
            else:
                offset = self.constants['RKName'] + 4
            h = np.memmap(self.filename, mode='r', offset=offset, shape=(1,),
                dtype=np.dtype([
                    ('initFlags', 'uint32'),
                    ('pulseCapacity', 'uint32'),
                    ('pulseToRayRatio', 'uint16'),
                    ('doNotUse', 'uint16'),
                    ('healthNodeCount', 'uint32'),
                    ('healthBufferDepth', 'uint32'),
                    ('statusBufferDepth', 'uint32'),
                    ('configBufferDepth', 'uint32'),
                    ('positionBufferDepth', 'uint32'),
                    ('pulseBufferDepth', 'uint32'),
                    ('rayBufferDepth', 'uint32'),
                    ('productBufferDepth', 'uint32'),
                    ('controlCapacity', 'uint32'),
                    ('waveformCalibrationCapacity', 'uint32'),
                    ('healthNodeBufferSize', 'uint64'),
                    ('healthBufferSize', 'uint64'),
                    ('statusBufferSize', 'uint64'),
                    ('configBufferSize', 'uint64'),
                    ('positionBufferSize', 'uint64'),
                    ('pulseBufferSize', 'uint64'),
                    ('rayBufferSize', 'uint64'),
                    ('productBufferSize', 'uint64'),
                    ('pulseSmoothFactor', 'uint32'),
                    ('pulseTicsPerSecond', 'uint32'),
                    ('positionSmoothFactor', 'uint32'),
                    ('positionTicsPerSecond', 'uint32'),
                    ('positionLatency', 'f8'),
                    ('latitude', 'f8'),
                    ('longitude', 'f8'), 
                    ('heading', 'f4'),
                    ('radarHeight', 'f4'),
                    ('wavelength', 'f4'),
                    ('name_raw', 'uint8', (self.constants['RKName'],)),
                    ('filePrefix_raw', 'uint8', (self.constants['RKMaximumPrefixLength'],)),
                    ('dataPath_raw', 'uint8', (self.constants['RKMaximumFolderPathLength'],))
                ])
            )
        elif self.header['buildNo'] == 1:
            h = np.memmap(self.filename, mode='r',
                offset=self.constants['RKName'] + 4, shape=(1,),
                dtype=np.dtype([
                    ('initFlags', 'uint32'),
                    ('pulseCapacity', 'uint32'),
                    ('pulseToRayRatio', 'uint32'),
                    ('healthNodeCount', 'uint32'),
                    ('configBufferDepth', 'uint32'),
                    ('positionBufferDepth', 'uint32'),
                    ('pulseBufferDepth', 'uint32'),
                    ('rayBufferDepth', 'uint32'),
                    ('controlCount', 'uint32'),
                    ('latitude', 'f8'),
                    ('longitude', 'f8'), 
                    ('heading', 'f4'),
                    ('radarHeight', 'f4'),
                    ('wavelength', 'f4'),
                    ('name_raw', 'uint8', (self.constants['RKName'],)),
                    ('filePrefix_raw', 'uint8', (self.constants['RKMaximumPrefixLength'],)),
                    ('dataPath_raw', 'uint8', (self.constants['RKMaximumFolderPathLength'],))
                ])
            )
        self.header['desc'] = {field: h[0][field] for field in h[0].dtype.names}
        self.header['desc']['name'] = ''\
            .join(chr(num) for num in self.header['desc']['name_raw'])\
            .replace('\x00', ' ').strip()
        self.header['desc']['filePrefix'] = ''\
            .join(chr(num) for num in self.header['desc']['filePrefix_raw'])\
            .replace('\x00', ' ').strip()
        self.header['desc']['dataPath'] = ''\
            .join(chr(num) for num in self.header['desc']['dataPath_raw'])\
            .replace('\x00', ' ').strip()
        
        if not (posFilename is None):
            posOverrideFields = ["latitude", "longitude", "heading"]
            with open(posFilename, "r") as posFile:
                for posField in posOverrideFields:
                    line = posFile.readline()
                    if not line:
                        break
                    val = float(line.strip())
                    self.header['desc'][posField] = val
        #-------------------------------------------------------------------------------------------
        
        
        #------Get config and waveforms-------------------------------------------------------------
        #Build 8 config
        if self.header['buildNo'] == 8:
            offset = self.constants['RKRadarDescOffset'] +\
                self.constants['RKRadarDesc']
            c = np.memmap(self.filename, mode='r', offset=offset, shape=(1,),
                dtype=np.dtype([
                    ('i', 'uint64'),
                    ('volumeIndex', 'uint64'),
                    ('sweepIndex', 'uint64'),
                    ('sweepElevation', 'f4'),
                    ('sweepAzimuth', 'f4'),
                    ('startMarker', 'uint32'),
                    ('prt', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('pw', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('pulseGateCount', 'uint32'),
                    ('pulseGateSize', 'f4'),
                    ('transitionGateCount', 'uint32'),
                    ('ringFilterGateCount', 'uint32'),
                    ('waveformId', 'uint32', (self.constants['RKMaxFilterCount'],)),
                    ('noise', 'f4', (2,)),
                    ('systemZCal', 'f4', (2,)),
                    ('systemDCal', 'f4'),
                    ('systemPCal', 'f4'),
                    ('ZCal', 'f4', (2,self.constants['RKMaxFilterCount'])),
                    ('DCal', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('PCal', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('SNRThreshold', 'f4'),
                    ('SQIThreshold', 'f4'),
                    ('waveformName', 'uint8', (self.constants['RKName'],)),
                    ('trash', 'uint64', (3,)),
                    ('momentMethod', 'uint8'),
                    ('userIntegerParameters', 'uint32', (8,)),
                    ('userFloatParameters', 'f4', (8,)),
                    ('vcpDefinition', 'uint8', (480,))
                ])
            )
            config = {field: c[0][field] for field in c[0].dtype.names}
            config['ZCal'] = config['ZCal'].T
            config['waveformName'] = ''\
                .join(chr(num) for num in config['waveformName'])\
                .replace('\x00', ' ').strip()
            config['vcpDefinition'] = ''\
                .join(chr(num) for num in config['vcpDefinition'])\
                .replace('\x00', ' ').strip()
            del config['trash']
            self.header['config'] = config
            
        
        #Build 6/7 config
        if self.header['buildNo'] == 6 or self.header['buildNo'] == 7:
            offset = self.constants['RKRadarDescOffset'] +\
                self.constants['RKRadarDesc']
            c = np.memmap(self.filename, mode='r', offset=offset, shape=(1,),
                dtype=np.dtype([
                    ('i', 'uint64'),
                    ('sweepElevation', 'f4'),
                    ('sweepAzimuth', 'f4'),
                    ('startMarker', 'uint32'),
                    ('prt', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('pw', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('pulseGateCount', 'uint32'),
                    ('pulseGateSize', 'f4'),
                    ('transitionGateCount', 'uint32'),
                    ('ringFilterGateCount', 'uint32'),
                    ('waveformId', 'uint32', (self.constants['RKMaxFilterCount'],)),
                    ('noise', 'f4', (2,)),
                    ('systemZCal', 'f4', (2,)),
                    ('systemDCal', 'f4'),
                    ('systemPCal', 'f4'),
                    ('ZCal', 'f4', (2,self.constants['RKMaxFilterCount'])),
                    ('DCal', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('PCal', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('SNRThreshold', 'f4'),
                    ('SQIThreshold', 'f4'),
                    ('waveformName', 'uint8', (self.constants['RKName'],)),
                    ('trash', 'uint64', (2,)),
                    ('momentMethod', 'uint8'),
                    ('vcpDefinition', 'uint8', (512,))
                ])
            )
            config = {field: c[0][field] for field in c[0].dtype.names}
            config['ZCal'] = config['ZCal'].T
            config['waveformName'] = ''\
                .join(chr(num) for num in config['waveformName'])\
                .replace('\x00', ' ').strip()
            config['vcpDefinition'] = ''\
                .join(chr(num) for num in config['vcpDefinition'])\
                .replace('\x00', ' ').strip()
            del config['trash']
            self.header['config'] = config
            
        #Build 6/7/8 Waveforms
        if self.header['buildNo'] >= 6 and self.header['buildNo'] <= 8:
            offset = self.constants['RKFileHeader']
            w = np.memmap(self.filename, mode='r', offset=offset, shape=(1,),
                dtype=np.dtype([
                    ('count', 'uint8'),
                    ('depth', 'uint32'),
                    ('type', 'uint32'),
                    ('name', 'uint8', (128,)),
                    ('fc', 'f8'),
                    ('fs', 'f8'),
                    ('filterCounts', 'uint8', (self.constants['RKMaximumWaveformCount'],))
                ])
            )
            self.header['waveform'] =\
                {field: w[0][field] for field in w[0].dtype.names}
            self.header['waveform']['filterCounts'] =\
                self.header['waveform']['filterCounts'][0:self.header['waveform']['count']]
            self.header['waveform']['name'] = ''\
                .join(chr(num) for num in self.header['waveform']['name'])\
                .replace('\x00', ' ').strip()
                
            offset += self.constants['RKWaveFileGlobalHeader']
            filters = []
            tones = []
            for i in range(self.header['waveform']['count']):
                tmp = []
                for j in range(self.header['waveform']['filterCounts'][i]):
                    w = np.memmap(self.filename, mode='r', offset=offset,
                        shape=(self.header['waveform']['filterCounts'][i],),
                        dtype=np.dtype([
                            ('name', 'uint32'),
                            ('origin', 'uint32'),
                            ('length', 'uint32'),
                            ('inputOrigin', 'uint32'),
                            ('outputOrigin', 'uint32'),
                            ('maxDataLength', 'uint32'),
                            ('subCarrierFrequency', 'f4'),
                            ('sensitivityGain', 'f4'),
                            ('filterGain', 'f4'),
                            ('fullScale', 'f4'),
                            ('lowerBoundFrequency', 'f4'),
                            ('upperBoundFrequency', 'f4'),
                            ('padding', 'uint8', (16,))
                        ])
                    )
                    for filter in w:
                        filter = {field: filter[field] for field in filter.dtype.names}
                        del filter['padding']
                        tmp.append(filter)
                    offset += self.constants['RKFilterAnchorSize']
                filters += tmp
                
                depth = self.header['waveform']['depth']
                w2 = np.memmap(self.filename, mode='r', offset=offset, shape=(1,),
                    dtype=np.dtype([
                        ('samples', 'f4', (depth,2)),
                        ('iSamples', 'int16', (depth,2))
                    ])
                )
                offset += 2 * depth * (4 + 2)
                x = w2[0]['samples']
                y = w2[0]['iSamples']
                gsamp = {
                    'samples': x[:,0] + 1j*x[:,1],
                    'iSamples': y[:,0] + 1j*y[:,1],  
                }
                tones += [gsamp]
            if len(filters) == 1:
                filters = filters[0]
            if len(tones) == 1:
                tones = tones[0]
            self.header['waveform']['filters'] = filters
            self.header['waveform']['tones'] = tones
            self.header['config']['pw'] =\
                self.header['config']['pw'][self.header['waveform']['filterCounts'][0]]
            self.header['config']['prt'] =\
                self.header['config']['prt'][self.header['config']['prt'] > 0]
            if len(self.header['config']['prt']) == 1:
                self.header['config']['prt'] = self.header['config']['prt'][0]
                
        #Build 5 Config and Waveforms
        elif self.header['buildNo'] == 5:
            offset = self['constants']['RKName'] + 4 + self['constants']['RKRadarDesc']
            c = np.memmap(self.filename, mode='r', offset=offset, shape=(1,),
                dtype=np.dtype([
                    ('i', 'uint64'),
                    ('sweepElevation', 'f4'),
                    ('sweepAzimuth', 'f4'),
                    ('startMarker', 'uint32'),
                    ('filterCount', 'uint8')
                ])
            )
            offset += 21
            c2 = np.memmap(self.filename, mode='r', offset=offset,
                shape=(self.constants['RKMaxFilterCount'],),
                dtype=np.dtype([
                    ('name', 'uint32'),
                    ('origin', 'uint32'),
                    ('length', 'uint32'),
                    ('inputOrigin', 'uint32'),
                    ('outputOrigin', 'uint32'),
                    ('maxDataLength', 'uint32'),
                    ('subCarrierFrequency', 'f4'),
                    ('sensitivityGain', 'f4'),
                    ('filterGain', 'f4'),
                    ('fullScale', 'f4'),
                    ('lowerBoundFrequency', 'f4'),
                    ('upperBoundFrequency', 'f4'),
                    ('padding', 'uint8', (16,))
                ])
            )
            offset += self.constants['RKMaxFilterCount'] * self.constants['RKFilterAnchorSize']
            c3 = np.memmap(self.filename, mode='r', offset=offset, shape=(1,),
                dtype = np.dtype([
                    ('prt', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('pw', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('pulseGateCount', 'uint32'),
                    ('pulseGateSize', 'f4'),
                    ('pulseRingFilterGateCount', 'uint32'),
                    ('waveformId', 'uint32', (self.constants['RKMaxFilterCount'],)),
                    ('noise', 'f4', (2,)),
                    ('systemZCal', 'f4', (2,)),
                    ('systemDCal', 'f4'),
                    ('systemPCal', 'f4'),
                    ('ZCal', 'f4', (2,self.constants['RKMaxFilterCount'])),
                    ('DCal', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('PCal', 'f4'),
                    ('SNRThreshold', 'f4'),
                    ('SQIThreshold', 'f4'),
                    ('waveform_raw', 'uint8', (self.constants['RKName'],)),
                    ('vcpDefinition_raw', 'uint8', (self.constants['RKName'],))
                ])
            )
            self.header['config'] = {field: c[0][field] for field in c[0].dtype.names}
            self.header['config']['filterAnchors'] = []
            for filterAnchor in c2:
                filterAnchor = {field: filterAnchor[field] for field in filterAnchor.dtype.names}
                self.header['config']['filterAnchors'] += [filterAnchor]
            if len(self.header['config']['filterAnchors']) == 1:
                self.header['config']['filterAnchors'] = self.header['config']['filterAnchors'][0]
            for field in c3[0].dtype.names[0:-2]:
                self.header['config'][field] = c3[0][field]
            self.header['config']['ZCal'] = self.header['config']['ZCal'].T
            self.header['config']['waveform'] = ''\
                .join(chr(num) for num in c3[0]['waveform_raw'])\
                .replace('\x00', ' ').strip()
            self.header['config']['vcpDefinition'] = ''\
                .join(chr(num) for num in c3[0]['vcpDefinition_raw'])\
                .replace('\x00', ' ').strip()
            offset = self.constants['RKName'] + 4 +\
                self.constants['RKRadarDesc'] + self.constants['RKConfigV1']
            file.seek(offset, 0)
            self.header['dataType'] = struct.unpack('B', file.read(1))[0]
            offset = self.constants['RKFileHeader']
            self.header['config']['pw'] =\
                self.header['config']['pw'](self.header['config']['filterCount'])
            self.header['config']['prt'] =\
                self.header['config']['prt'][self.header['config']['prt'] > 0]
            if len(self.header['config']['prt']) == 1:
                self.header['config']['prt'] = self.header['config']['prt'][0]
                
        #Build 2-4 Config and Waveforms
        elif self.header['buildNo'] >= 2 and self.header['buildNo'] < 5:
            offset = self.constants['RKName'] + 4 + self.constants['RKRadarDesc']
            c = np.memmap(self.filename, mode='r', offset=offset, shape=(1,),
                dtype = np.dtype([
                    ('i', 'uint64'),
                    ('sweepElevation', 'f4'),
                    ('sweepAzimuth', 'f4'),
                    ('startMarker', 'uint32'),
                    ('filterCount', 'uint8'),
                ])
            )
            offset += 21
            c2 = np.memmap(self.filename, mode='r', offset=offset,
                shape=(self.constants['RKMaxFilterCount'],),
                dtype=np.dtype([
                    ('name', 'uint32'),
                    ('origin', 'uint32'),
                    ('length', 'uint32'),
                    ('inputOrigin', 'uint32'),
                    ('outputOrigin', 'uint32'),
                    ('maxDataLength', 'uint32'),
                    ('subCarrierFrequency', 'f4'),
                    ('sensitivityGain', 'f4'),
                    ('filterGain', 'f4'),
                    ('fullScale', 'f4'),
                    ('lowerBoundFrequency', 'f4'),
                    ('upperBoundFrequency', 'f4'),
                    ('padding', 'uint8', (16,))
                ])
            )
            offset += self.constants['RKMaxFilterCount'] * self.constants['RKFilterAnchorSize']
            c3 = np.memmap(self.filename, mode='r', offset=offset, shape=(1,),
                dtype = np.dtype([
                    ('prt', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('pw', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('pulseGateCount', 'uint32'),
                    ('pulseGateSize', 'f4'),
                    ('pulseRingFilterGateCount', 'uint32'),
                    ('waveformId', 'uint32', (self.constants['RKMaxFilterCount'],)),
                    ('noise', 'f4', (2,)),
                    ('systemZCal', 'f4', (2,)),
                    ('systemDCal', 'f4'),
                    ('systemPCal', 'f4'),
                    ('ZCal', 'f4', (2,self.constants['RKMaxFilterCount'])),
                    ('DCal', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('PCal', 'f4'),
                    ('SNRThreshold', 'f4'),
                    ('waveform_raw', 'uint8', (self.constants['RKName'],)),
                    ('vcpDefinition_raw', 'uint8', (self.constants['RKName'],))
                ])
            )
            self.header['config'] = {field: c[0][field] for field in c[0].dtype.names}
            self.header['config']['filterAnchors'] = []
            for filterAnchor in c2:
                filterAnchor = {field: filterAnchor[field] for field in filterAnchor.dtype.names}
                self.header['config']['filterAnchors'] += [filterAnchor]
            if len(self.header['config']['filterAnchors']) == 1:
                self.header['config']['filterAnchors'] = self.header['config']['filterAnchors'][0]
            for field in c3[0].dtype.names[0:-2]:
                self.header['config'][field] = c3[0][field]
            self.header['config']['ZCal'] = self.header['config']['ZCal'].T
            self.header['config']['waveformName'] = ''\
                .join(chr(num) for num in c3[0]['waveform_raw'])\
                .replace('\x00', ' ').strip()
            self.header['config']['vcpDefinition'] = ''\
                .join(chr(num) for num in c3[0]['vcpDefinition_raw'])\
                .replace('\x00', ' ').strip()
            self.header["dataType"] = 1
            offset = self.constants['RKFileHeader']
            
        #Build 1 Config and Waveforms
        elif self.header['buildNo'] == 1:
            c = np.memmap(self.filename, mode='r',
                offset=self.constants['RKName'] + 4 + self.constants['RKRadarDesc'], shape=(1,),
                dtype = np.dtype([
                    ('i', 'uint64'),
                    ('pw', 'uint32', (self.constants['RKMaxFilterCount'],)),
                    ('prf', 'uint32', (self.constants['RKMaxFilterCount'],)),
                    ('gateCount', 'uint32', (self.constants['RKMaxFilterCount'],)),
                    ('waveformId', 'uint32', (self.constants['RKMaxFilterCount'],)),
                    ('noise', 'f4', (2,)),
                    ('ZCal', 'f4', (2,self.constants['RKMaxFilterCount'])),
                    ('DCal', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('PCal', 'f4', (self.constants['RKMaxFilterCount'],)),
                    ('SNRThreshold', 'f4'),
                    ('sweepElevation', 'f4'),
                    ('sweepAzimuth', 'f4'),
                    ('startMarker', 'uint32'),
                    ('waveform_name_raw', 'uint8', (self.constants['RKName'],)),
                    ('vcpDefinition_raw', 'uint8', (self.constants['RKName'],))
                ])
            )
            self.header['config'] = {field: c[0][field] for field in c[0].dtype.names}
            self.header['config']['waveformName'] = ''\
                .join(chr(num) for num in self.header['config']['waveform_raw'])\
                .replace('\x00', ' ').strip()
            self.header['config']['vcpDefinition'] = ''\
                .join(chr(num) for num in self.header['config']['vcpDefinition_raw'])\
                .replace('\x00', ' ').strip()
            self.header['dataType'] = 1
            offset = self.constants['RKFileHeader']
        #-------------------------------------------------------------------------------------------
        
        
        #------Read Pulses--------------------------------------------------------------------------
        #Partially read first pulse
        file.seek(offset+28, 0)
        pulseStart = struct.unpack('III', file.read(12))
        capacity = pulseStart[0]
        gateCount = pulseStart[1]
        downSampledGateCount = pulseStart[2]
        pulseDataSize = file.seek(0, 2) - offset
        if verbose:
            print(f"Pulse data size: {pulseDataSize}")
        file.close()
        if verbose:
            print(f"gateCount = {gateCount}   capacity = {capacity} "
                f"  downSampledGateCount = {downSampledGateCount}")
        
        #Dimensions
        if verbose:
            print(f"data offset = {offset}")
            if maxPulse != None:
                print(f"Reading {maxPulse} pulses ...")
            else:
                print("Reading pulses ...")
        
        #Read all pulses
        if self.header['buildNo'] == 7 or self.header['buildNo'] == 8:
            if self.header['dataType'] == 1:
                #Raw I/Q straight from the transceiver
                IQDtype = np.dtype([
                    ('i', 'uint64'),
                    ('n', 'uint64'),
                    ('t', 'uint64'),
                    ('s', 'uint32'),
                    ('capacity', 'uint32'),
                    ('gateCount', 'uint32'),
                    ('downSampledGateCount', 'uint32'),
                    ('marker', 'uint32'),
                    ('pulseWidthSampleCount', 'uint32'),
                    ('time_tv_sec', 'uint64'),
                    ('time_tv_usec', 'uint64'),
                    ('timeDouble', 'f8'),
                    ('rawAzimuth', 'uint8', (4,)),
                    ('rawElevation', 'uint8', (4,)),
                    ('configIndex', 'uint16'),
                    ('configSubIndex', 'uint16'),
                    ('positionIndex', 'uint32'),
                    ('gateSizeMeters', 'f4'),
                    ('elevationDegrees', 'f4'),
                    ('azimuthDegrees', 'f4'),
                    ('elevationVelocityDegreesPerSecond', 'f4'),
                    ('azimuthVelocityDegreesPerSecond', 'f4'),
                    ('padding', 'uint8', (84,)),
                    ('iq', 'int16', (2, gateCount, 2))
                ])
                numPulses = pulseDataSize // IQDtype.itemsize
                if verbose:
                    print(f"Number of pulses: {numPulses}")
                m = np.memmap(self.filename, mode='r', offset=offset, 
                    shape=(numPulses,) if maxPulse == None else (maxPulse,), dtype = IQDtype
                )
            else:
                #Compressed I/Q
                IQDtype = np.dtype([
                    ('i', 'uint64'),
                    ('n', 'uint64'),
                    ('t', 'uint64'),
                    ('s', 'uint32'),
                    ('capacity', 'uint32'),
                    ('gateCount', 'uint32'),
                    ('downSampledGateCount', 'uint32'),
                    ('marker', 'uint32'),
                    ('pulseWidthSampleCount', 'uint32'),
                    ('time_tv_sec', 'uint64'),
                    ('time_tv_usec', 'uint64'),
                    ('timeDouble', 'f8'),
                    ('rawAzimuth', 'uint8', (4,)),
                    ('rawElevation', 'uint8', (4,)),
                    ('configIndex', 'uint16'),
                    ('configSubIndex', 'uint16'),
                    ('positionIndex', 'uint32'),
                    ('gateSizeMeters', 'f4'),
                    ('elevationDegrees', 'f4'),
                    ('azimuthDegrees', 'f4'),
                    ('elevationVelocityDegreesPerSecond', 'f4'),
                    ('azimuthVelocityDegreesPerSecond', 'f4'),
                    ('padding', 'uint8', (84,)),
                    ('iq', 'f4', (2, downSampledGateCount, 2))
                ])
                numPulses = pulseDataSize // IQDtype.itemsize
                if verbose:
                    print(f"Number of pulses: {numPulses}")
                m = np.memmap(self.filename, mode='r', offset=offset, 
                    shape=(numPulses,) if maxPulse == None else (maxPulse,), dtype = IQDtype
                )
        else:
            if self.header['dataType'] == 1:
                #Raw I/Q straight from the transceiver
                IQDtype = np.dtype([
                    ('i', 'uint64'),
                    ('n', 'uint64'),
                    ('t', 'uint64'),
                    ('s', 'uint32'),
                    ('capacity', 'uint32'),
                    ('gateCount', 'uint32'),
                    ('downSampledGateCount', 'uint32'),
                    ('marker', 'uint32'),
                    ('pulseWidthSampleCount', 'uint32'),
                    ('time_tv_sec', 'uint64'),
                    ('time_tv_usec', 'uint64'),
                    ('timeDouble', 'f8'),
                    ('rawAzimuth', 'uint8', (4,)),
                    ('rawElevation', 'uint8', (4,)),
                    ('configIndex', 'uint16'),
                    ('configSubIndex', 'uint16'),
                    ('azimuthBinIndex', 'uint16'),
                    ('gateSizeMeters', 'f4'),
                    ('elevationDegrees', 'f4'),
                    ('azimuthDegrees', 'f4'),
                    ('elevationVelocityDegreesPerSecond', 'f4'),
                    ('azimuthVelocityDegreesPerSecond', 'f4'),
                    ('iq', 'int16', (2, gateCount, 2))
                ])
                numPulses = pulseDataSize // IQDtype.itemsize
                if verbose:
                    print(f"Number of pulses: {numPulses}")
                m = np.memmap(self.filename, mode='r', offset=offset, 
                    shape=(numPulses,) if maxPulse == None else (maxPulse,), dtype = IQDtype
                )
            else:
                #Compressed I/Q (non build 7)
                IQDtype = np.dtype([
                    ('i', 'uint64'),
                    ('n', 'uint64'),
                    ('t', 'uint64'),
                    ('s', 'uint32'),
                    ('capacity', 'uint32'),
                    ('gateCount', 'uint32'),
                    ('downSampledGateCount', 'uint32'),
                    ('marker', 'uint32'),
                    ('pulseWidthSampleCount', 'uint32'),
                    ('time_tv_sec', 'uint64'),
                    ('time_tv_usec', 'uint64'),
                    ('timeDouble', 'f8'),
                    ('rawAzimuth', 'uint8', (4,)),
                    ('rawElevation', 'uint8', (4,)),
                    ('configIndex', 'uint16'),
                    ('configSubIndex', 'uint16'),
                    ('azimuthBinIndex', 'uint16'),
                    ('gateSizeMeters', 'f4'),
                    ('elevationDegrees', 'f4'),
                    ('azimuthDegrees', 'f4'),
                    ('elevationVelocityDegreesPerSecond', 'f4'),
                    ('azimuthVelocityDegreesPerSecond', 'f4'),
                    ('iq', 'f4', (2,downSampledGateCount,2)),
                ])
                numPulses = pulseDataSize // IQDtype.itemsize
                if verbose:
                    print(f"Number of pulses: {numPulses}")
                m = np.memmap(self.filename, mode='r', offset=offset, 
                    shape=(numPulses,) if maxPulse == None else (maxPulse,), dtype = IQDtype
                )
        self.pulses = np.array(m)
        self.pulses['iq'] = self.pulses['iq'].transpose(0, 3, 2, 1)
        #-------------------------------------------------------------------------------------------
        
        
        #------Set dataType as string and see if waveform is recorded-------------------------------
        if self.header['dataType'] == 1:
            dt = 'raw'
        elif self.header['dataType'] == 2:
            dt = 'compressed'
        else:
            dt = 'unknown'
        self.header['dataType'] = dt
        
        if (not ('waveform' in self.header))\
            or (type(self.header['waveform']) == str and len(self.header['waveform']) == 0):
            self.header['waveform'] = 'not recorded'
        
    def pulseToDict(self, idx: int):
        return {field: self.pulses[idx][field] for field in self.pulses[idx].dtype.names}
    
    def elArray(self):
        return self.pulses['elevationDegrees']
    
    def azArray(self):
        return self.pulses['azimuthDegrees']
    


# ----------------------------------------------------------------------------------------------------
# moment generation
# ----------------------------------------------------------------------------------------------------

# Ameya's
def ccf(X1, X2=None, lag=0):
	if X2 is None:
		X2 = X1 # acf
	if X1.shape != X2.shape:
		raise ValueError("Two array shapes not equal.")

	shape = X1.shape
	axis_to_avg = len(shape)-1

	padding = np.zeros(shape=tuple(
		(shape[i] if i < axis_to_avg else np.abs(lag))
		for i in range(len(shape))))

	if lag >= 0:
		prod = X1[lag:] * np.conjugate(X2[:(None if lag == 0 else -lag)])
	else:
		prod = X1[:lag] * np.conjugate(X2[-lag:])

	return np.mean(np.concatenate([prod, padding], axis=axis_to_avg), axis=axis_to_avg)

"""def ccf(x1, x2, l):
	summ = 0 + 0j
	M = x1.size
	if not x1.size == x2.size:
		print("Timeseries sizes different")
	for m in range(0, M-np.abs(l)):
		summ = summ + np.dot(x1[m],np.conj(x2[m-l]))
	return summ/(M-np.abs(l)-1)"""

def acf(x, l):
	return ccf(x, x, l)

def get_moments(X_ho, X_vo, N_h, N_v, R, va, C, Cd, Cp):
	# X_h and X_v have dims (range, ray, pulse)

	# C = C[0][0]
	# Cd = Cd[0]
	# Cp = Cp[0]

	if not np.isfinite(N_h) or N_h < 0:
		N_h = 0
	if not np.isfinite(N_v) or N_v < 0:
		N_v = 0

	C = np.nanmean(C[:,0])
	Cd = np.nanmean(Cd)
	Cp = np.nanmean(Cp)

	moments = {'DBZ': None, 'VEL': None, 'WIDTH': None, 'ZDR': None, 'RHOHV': None, 'PHIDP': None, 'SNRH': None, 'SNRV': None}

	DBZOUT = np.full((X_ho.shape[1], X_ho.shape[0]), np.nan)
	VELOUT, WIDTHOUT, ZDROUT, RHOHVOUT, PHIDPOUT, SNRHOUT, SNRVOUT = DBZOUT.copy(), DBZOUT.copy(), DBZOUT.copy(), DBZOUT.copy(), DBZOUT.copy(), DBZOUT.copy(), DBZOUT.copy()

	for it in range(X_ho.shape[1]):
		for ir in range(X_ho.shape[0]):
			X_h = X_ho[ir,it]
			X_v = X_vo[ir,it]
			r = R[ir]

			P_h = np.real(acf(X_h, 0)) # Rxx(V,0)
			P_v = np.real(acf(X_v, 0))
			S_h = P_h - N_h
			S_v = P_v - N_v

			cross = ccf(X_h, X_v, 0)
			lag1h = acf(X_h,1)
			# print(P_h, N_h, P_v, N_v)

			DBZOUT[it,ir] = 10*np.log10(S_h) + 20*np.log10(r if not r==0 else 1e-10) + 10*np.log10(C if not C==0 else 1e-10)
			VELOUT[it,ir] = (va/np.pi)*np.angle(lag1h)
			WIDTHOUT[it,ir] = (np.sqrt(2)*va/np.pi)*np.sqrt(np.abs(np.log(S_h/np.abs(lag1h))))
			ZDROUT[it,ir] = 10*np.log10(S_h/S_v) # + 10*np.log10(Cd)
			RHOHVOUT[it,ir] = np.abs(cross)/np.sqrt(S_h*S_v)
			PHIDPOUT[it,ir] = np.rad2deg(np.angle(cross)) + Cp # np.atan2(np.imag(S_h),np.real(S_h))-np.atan2(np.imag(S_v),np.real(S_v)) + Cp
			SNRHOUT[it,ir] = 10*np.log10(S_h/N_h)
			SNRVOUT[it,ir] = 10*np.log10(S_v/N_v)

	PHIDPOUT[PHIDPOUT < -180] += 360
	PHIDPOUT[PHIDPOUT > 180] -= 360

	moments['DBZ'] = DBZOUT
	moments['VEL'] = VELOUT
	moments['WIDTH'] = WIDTHOUT
	moments['ZDR'] = ZDROUT
	moments['RHOHV'] = RHOHVOUT
	moments['PHIDP'] = PHIDPOUT
	moments['SNRH'] = SNRHOUT
	moments['SNRV'] = SNRVOUT

	return moments



# ----------------------------------------------------------------------------------------------------
# RaXPol formatting
# ----------------------------------------------------------------------------------------------------

class raxpolCf:
    #------Start Required Functions-----------------------------------------------------------------
    def setVolume(self, volNum: int = 0):
        self.variables["volume_number"]["data"] = volNum
        self.requiredBools["volume"] = True
        
    def setSweep(self, sweepNum: int = 0):
        self.variables["sweep_number"]["data"] = np.array([sweepNum], dtype=np.int32)
        self.requiredBools["sweep"] = True
        
    def setTime(self, unixTimeArr: NDArray[np.float64], time_zone: str = 'zulu'):
        if (unixTimeArr.dtype != np.float64):
            raise TypeError("Expected array of np.float64")
        
        startTime = unixTimeArr[0]
        endTime = unixTimeArr[-1]
        
        timeVar = unixTimeArr - startTime
        nRays = len(timeVar)
        
        self.dimensions['time'] = nRays
        
        startTimeStr = datetime.fromtimestamp(startTime, tz=pytz.timezone(time_zone))\
            .astimezone(pytz.utc).isoformat()
        endTimeStr = datetime.fromtimestamp(endTime, tz=pytz.timezone(time_zone))\
            .astimezone(pytz.utc).isoformat()
            
        self.rootAttrs["time_coverage_start"] = startTimeStr.replace('+00:00', 'Z')
        self.rootAttrs["start_datetime"] = startTimeStr
        self.rootAttrs["time_coverage_end"] = endTimeStr.replace('+00:00', 'Z')
        self.rootAttrs["end_datetime"] = endTimeStr
        
        paddedStartTime = startTimeStr.replace('+00:00', 'Z') +\
            (self.dimensions["string_length_32"] - len(startTimeStr.replace('+00:00', 'Z')))*' '
        paddedEndTime = startTimeStr.replace('+00:00', 'Z') +\
            (self.dimensions["string_length_32"] - len(startTimeStr.replace('+00:00', 'Z')))*' '
            
        self.variables["time_coverage_start"]["data"] =\
            np.array([c for c in paddedStartTime], dtype="|S1")
        self.variables["time_coverage_end"]["data"] =\
            np.array([c for c in paddedEndTime], dtype="|S1")
            
        self.variables["time"]["units"] = "seconds since " + startTimeStr.replace('+00:00', 'Z')
        self.variables["time"]["data"] = np.ma.masked_invalid(timeVar)
        
        self.variables["sweep_start_ray_index"]["data"] = np.array([0], dtype=np.int32)
        self.variables["sweep_end_ray_index"]["data"] = np.array([nRays-1], dtype=np.int32)
        
        self.requiredBools["time"] = True
        
    def setRange(self, rangeGates: NDArray[np.float32]):
        if (rangeGates.dtype != np.float32):
            raise TypeError("Expected array of np.float32")
        
        nGates = len(rangeGates)
        firstGate = np.rint(rangeGates[0])
        dG = np.rint(rangeGates[1]-rangeGates[0])
        
        self.dimensions["range"] = nGates
        
        self.variables["range"]["meters_to_center_of_first_gate"] = str(firstGate)
        self.variables["range"]["meters_between_gates"] = str(dG)
        
        self.variables["range"]["data"] = np.ma.masked_invalid(rangeGates)
        
        self.requiredBools["range"] = True
        
    def setPosition(self, lat: float, lon: float):
        if lat < -90 or lat > 90:
            raise ValueError(f'Latitude {lat} out of -90 to 90 deg range.')
        if lon < -180 or lon > 180:
            raise ValueError(f'Longitude {lon} out of -180 to 180 deg range.')
        
        self.variables["latitude"]["data"] = np.ma.masked_invalid(lat)
        self.variables["longitude"]["data"] = np.ma.masked_invalid(lon)
        
        self.requiredBools["position"] = True
        
    def setScanningStrategy(self, strategy: str):
        if strategy == "ppi":
            self.variables["sweep_mode"]["data"] =\
                np.array([[c for c in 'azimuth_surveillance            ']], dtype='|S1')
            self.variables["fixed_angle"]["units"] = "elevation degrees"
        else:
            raise ValueError("Sorry, only ppi mode supported for now.")
        
        self.requiredBools["scanning_strategy"] = True
    
    def setTargetAngle(self, targetAngle: float):
        if not self.requiredBools["scanning_strategy"]:
            raise RuntimeError("Need to call setScanningStrategy() before this function.")
        if self.variables["fixed_angle"]["units"] == "elevation degrees":
            #ppi mode
            if targetAngle < -90 or targetAngle > 90:
                raise ValueError("Radar dish shouldn't be pointing into "
                                    "the floor or greater than vertical.")
            self.variables["fixed_angle"]["data"] =\
                np.ma.masked_invalid(np.array([targetAngle], dtype=np.float32))
        else:
            raise ValueError("Sorry, only ppi mode supported for now.")
        
        self.requiredBools["target_angle"] = True
    
    def setAzimuth(self, azimuths: NDArray[np.float32]):
        if (azimuths.dtype != np.float32):
            raise TypeError("Expected array of np.float32")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if len(azimuths) != self.dimensions["time"]:
            raise RuntimeError("Number of azimuths need to measure number "
                               "of rays from setTime() function call. "
                               f'For this file, that is {self.dimensions["time"]} rays.')
        
        self.variables["azimuth"]["data"] = np.ma.masked_invalid(azimuths)
        
        self.requiredBools["azimuth"] = True
        
    def setElevation(self, elevations: NDArray[np.float32]):
        if (elevations.dtype != np.float32):
            raise TypeError("Expected array of np.float32")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if len(elevations) != self.dimensions["time"]:
            raise RuntimeError("Number of elevations need to measure number "
                               "of rays from setTime() function call. "
                               f'For this file, that is {self.dimensions["time"]} rays.')
            
        self.variables["elevation"]["data"] = np.ma.masked_invalid(elevations)
        
        self.requiredBools["elevation"] = True
    
    def setPulseWidthSeconds(self, pulseWidths: NDArray[np.float32]):
        if (pulseWidths.dtype != np.float32):
            raise TypeError("Expected array of np.float32")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if len(pulseWidths) != self.dimensions["time"]:
            raise RuntimeError("Number of pulse widths need to measure number "
                               "of rays from setTime() function call. "
                               f'For this file, that is {self.dimensions["time"]} rays.')
            
        self.variables["pulse_width"]["data"] = np.ma.masked_invalid(pulseWidths)
        
        self.requiredBools["pulse_width"] = True
        
    def setPrtSeconds(self, pulse_repetition_times: NDArray[np.float32]):
        if (pulse_repetition_times.dtype != np.float32):
            raise TypeError("Expected array of np.float32")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if len(pulse_repetition_times) != self.dimensions["time"]:
            raise RuntimeError("Number of prt values need to measure number "
                               "of rays from setTime() function call. "
                               f'For this file, that is {self.dimensions["time"]} rays.')
            
        self.variables["prt"]["data"] = np.ma.masked_invalid(pulse_repetition_times)
        
        self.requiredBools["prt"] = True
        
    def setWavelengthMeters(self, wavelengths: NDArray[np.float32]):
        if (wavelengths.dtype != np.float32):
            raise TypeError("Expected array of np.float32")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if not self.requiredBools["prt"]:
            raise RuntimeError("Need to call setPrtSeconds() before this function, "
                               "for nyquist velocity calculation.")
        if len(wavelengths) != self.dimensions["time"]:
            raise RuntimeError("Number of wavelength values need to measure number "
                               "of rays from setTime() function call. "
                               f'For this file, that is {self.dimensions["time"]} rays.')
        
        self.variables["wavelength"]["data"] = wavelengths
        self.variables["nyquist_velocity"]["data"] =\
            np.ma.masked_invalid(0.25 * wavelengths / self.variables["prt"]["data"])
            
        self.requiredBools["wavelength"] = True
    #------End Required Functions-------------------------------------------------------------------
    
    
    #------Start Data Functions---------------------------------------------------------------------
    def setDBZ(self, DBZ: NDArray[np.float64]):
        if (DBZ.dtype != np.float64):
            raise TypeError("Expected array of np.float64")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if not self.requiredBools["range"]:
            raise RuntimeError("Need to call setRange() before this function.")
        if DBZ.shape != (self.dimensions["time"], self.dimensions["range"]):
            raise RuntimeError("Number of reflectivity values need to measure number "
                               "of rays and gates from setTime() abd setRange() function calls. "
                               f'For this file, that is {self.dimensions["time"]} rays, '
                               f'and {self.dimensions["range"]} gates.')
        
        self.variables["DBZ"]["data"] = np.ma.masked_invalid(DBZ)
        
        self.radarVarBools["DBZ"] = True
        
    def setVEL(self, VEL: NDArray[np.float64]):
        if (VEL.dtype != np.float64):
            raise TypeError("Expected array of np.float64")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if not self.requiredBools["range"]:
            raise RuntimeError("Need to call setRange() before this function.")
        if VEL.shape != (self.dimensions["time"], self.dimensions["range"]):
            raise RuntimeError("Number of velocity values need to measure number "
                               "of rays and gates from setTime() abd setRange() function calls. "
                               'For this file, that is {self.dimensions["time"]} rays, '
                               f'and {self.dimensions["range"]} gates.')
        
        self.variables["VEL"]["data"] = np.ma.masked_invalid(VEL)
        
        self.radarVarBools["VEL"] = True
        
    def setWIDTH(self, WIDTH: NDArray[np.float64]):
        if (WIDTH.dtype != np.float64):
            raise TypeError("Expected array of np.float64")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if not self.requiredBools["range"]:
            raise RuntimeError("Need to call setRange() before this function.")
        if WIDTH.shape != (self.dimensions["time"], self.dimensions["range"]):
            raise RuntimeError("Number of spectrum width values need to measure number "
                               "of rays and gates from setTime() abd setRange() function calls. "
                               f'For this file, that is {self.dimensions["time"]} rays, '
                               f'and {self.dimensions["range"]} gates.')
        
        self.variables["WIDTH"]["data"] = np.ma.masked_invalid(WIDTH)
        
        self.radarVarBools["WIDTH"] = True
        
    def setZDR(self, ZDR: NDArray[np.float64]):
        if (ZDR.dtype != np.float64):
            raise TypeError("Expected array of np.float64")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if not self.requiredBools["range"]:
            raise RuntimeError("Need to call setRange() before this function.")
        if ZDR.shape != (self.dimensions["time"], self.dimensions["range"]):
            raise RuntimeError("Number of differential reflectivity values need to measure number "
                               "of rays and gates from setTime() abd setRange() function calls. "
                               f'For this file, that is {self.dimensions["time"]} rays, '
                               f'and {self.dimensions["range"]} gates.')
            
        self.variables["ZDR"]["data"] = np.ma.masked_invalid(ZDR)
        
        self.radarVarBools["ZDR"] = True
        
    def setPHIDP(self, PHIDP: NDArray[np.float64], units: str):
        if (PHIDP.dtype != np.float64):
            raise TypeError("Expected array of np.float64")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if not self.requiredBools["range"]:
            raise RuntimeError("Need to call setRange() before this function.")
        if PHIDP.shape != (self.dimensions["time"], self.dimensions["range"]):
            raise RuntimeError("Number of differential phase values need to measure number "
                               "of rays and gates from setTime() abd setRange() function calls. "
                               f'For this file, that is {self.dimensions["time"]} rays, '
                               f'and {self.dimensions["range"]} gates.')
        if not (units == "degrees" or units == "radians"):
            raise ValueError("Units required, and can only be \"degres\" or \"radians\".")

        self.variables["PHIDP"]["units"] = units
        self.variables["PHIDP"]["data"] = np.ma.masked_invalid(PHIDP)
        
        self.radarVarBools["PHIDP"] = True
        
    def setRHOHV(self, RHOHV: NDArray[np.float64]):
        if (RHOHV.dtype != np.float64):
            raise TypeError("Expected array of np.float64")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if not self.requiredBools["range"]:
            raise RuntimeError("Need to call setRange() before this function.")
        if RHOHV.shape != (self.dimensions["time"], self.dimensions["range"]):
            raise RuntimeError("Number of correlation coefficient values need to measure number "
                               "of rays and gates from setTime() abd setRange() function calls. "
                               f'For this file, that is {self.dimensions["time"]} rays, '
                               f'and {self.dimensions["range"]} gates.')

        self.variables["RHOHV"]["data"] = np.ma.masked_invalid(RHOHV)
        
        self.radarVarBools["RHOHV"] = True
        
    def setSNRH(self, SNRH: NDArray[np.float64]):
        if (SNRH.dtype != np.float64):
            raise TypeError("Expected array of np.float64")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if not self.requiredBools["range"]:
            raise RuntimeError("Need to call setRange() before this function.")
        if SNRH.shape != (self.dimensions["time"], self.dimensions["range"]):
            raise RuntimeError("Number of signal to noise ratio values need to measure number "
                               "of rays and gates from setTime() abd setRange() function calls. "
                               f'For this file, that is {self.dimensions["time"]} rays, '
                               f'and {self.dimensions["range"]} gates.')

        self.variables["SNRH"]["data"] = np.ma.masked_invalid(SNRH)
        
        self.radarVarBools["SNRH"] = True

    def setSNRV(self, SNRV: NDArray[np.float64]):
        if (SNRV.dtype != np.float64):
            raise TypeError("Expected array of np.float64")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if not self.requiredBools["range"]:
            raise RuntimeError("Need to call setRange() before this function.")
        if SNRV.shape != (self.dimensions["time"], self.dimensions["range"]):
            raise RuntimeError("Number of signal to noise ratio values need to measure number "
                               "of rays and gates from setTime() abd setRange() function calls. "
                               f'For this file, that is {self.dimensions["time"]} rays, '
                               f'and {self.dimensions["range"]} gates.')

        self.variables["SNRV"]["data"] = np.ma.masked_invalid(SNRV)
        
        self.radarVarBools["SNRV"] = True
    #------End Data Functions-----------------------------------------------------------------------
    
    
    #------Begin Optional Data Functions------------------------------------------------------------
    def setPulseBoundaries(self, boundaries: NDArray[np.int32]):
        if (boundaries.dtype != np.int32):
            raise TypeError("Expected array of np.int32")
        if not self.requiredBools["time"]:
            raise RuntimeError("Need to call setTime() before this function.")
        if boundaries.shape != (self.dimensions["time"], self.dimensions["ray_start_end"]):
            raise RuntimeError("Number of start-end pairs need to match number of rays.")
        
        self.variables["pulse_boundaries"]["data"] = boundaries
        
        self.optionalVarBools["pulse_boundaries"] = True
    #-----------------------------------------------------------------------------------------------
    
    
    #------Begin Optional Documentation Functions---------------------------------------------------
    def setTitle(self, title: str):
        self.rootAttrs["title"] = title
        self.optionalBools["title"] = True
        
    def setHistory(self, history: str):
        self.rootAttrs["history"] = history
        self.optionalBools["history"] = True
        
    def setRadarTeam(self, team: str):
        self.rootAttrs["radar_team"] = team
        self.optionalBools["radar_team"] = True
        
    def setAddtlComments(self, comments: str):
        self.rootAttrs["comment"] = "Generated by raxpolCf.py "\
                                    "(Author: Ameya Naik, https://github.com/aeol1an). "\
                                    "Adapted from convert_px_cfrad23.m. " + comments
        self.optionalBools["addtl_comments"] = True
    #------End Optional Functions-------------------------------------------------------------------
    
    def saveToFile(self, filename: str):
        if not np.all(np.array([val for val in self.requiredBools.values()])):
            missing_functions = ""
            for key, val in self.requiredBools.items():
                if not val:
                    missing_functions += "\n" + key
            raise RuntimeError("All required functions need to be "
                               "called. Missing:" + missing_functions)
        if not np.any(np.array([val for val in self.radarVarBools.values()])):
            raise RuntimeError("At least one radar variable needs to be set.")
        
        file = netCDF4.Dataset(filename, 'w', format='NETCDF4')
        
        for key, value in self.rootAttrs.items():
            setattr(file, key, value)

        for key, value in self.dimensions.items():
            file.createDimension(key, value)
        
        #Define dict fields that are not just string attributes
        coreVarFields = ["type", "fill_value", "dims", "data"]
        for var in self.requiredVars:
            varDict = self.variables[var]
            ncvar = file.createVariable(var, varDict["type"], varDict["dims"], 
                                        fill_value=varDict["fill_value"])
            for key, val in varDict.items():
                if not key in coreVarFields:
                    setattr(ncvar, key, val)
            ncvar[:] = varDict["data"]
            
        for var, exists in self.optionalVarBools.items():
            if not exists:
                continue
            varDict = self.variables[var]
            ncvar = file.createVariable(var, varDict["type"], varDict["dims"], 
                                        fill_value=varDict["fill_value"])
            for key, val in varDict.items():
                if not key in coreVarFields:
                    setattr(ncvar, key, val)
            ncvar[:] = varDict["data"]
                
        for var, exists in self.radarVarBools.items():
            if not exists:
                continue
            varDict = self.variables[var]
            ncvar = file.createVariable(var, varDict["type"], varDict["dims"], 
                                        fill_value=varDict["fill_value"])
            for key, val in varDict.items():
                if not key in coreVarFields:
                    setattr(ncvar, key, val)
            ncvar[:] = varDict["data"]
            
        file.close()

    def __init__(self):
        self.requiredBools = {
            #All of these are required true
            "volume": False,
            "sweep": False,
            "time": False,
            "range": False,
            "position": False,
            "scanning_strategy": False,
            "target_angle": False,
            "azimuth": False,
            "elevation": False,
            "pulse_width": False,
            "prt": False,
            "wavelength": False
        }

        self.optionalBools = {
            #None of these are require true, but are helpful for documentation
            "title": False,
            "history": False,
            "radar_team": False,
            "addtl_comments": False,
        }
        
        self.rootAttrs = {
            "Conventions": "CF/Radial",
            "title": "",
            "institution": "University of Oklahoma",
            "references": "https://github.com/OURadar/RadarKit",
            "source": "RadarKit raw I/Q",
            "history": "",
            "comment": "Generated by raxpolCf.py (Author: Ameya Naik, https://github.com/aeol1an). "
                       "Adapted from convert_px_cfrad23.m.",
            "instrument_name": "RaXPol",
            "radar_team": "",
            "time_coverage_start": "tbd",
            "time_coverage_end": "tbd",
            "start_datetime": "tbd",
            "end_datetime": "tbd",
            "version": "CF-Radial-1.3",
        }
        
        self.dimensions = {
            "time": "tbd",
            "range": "tbd",
            "sweep": 1,
            "ray_start_end": 2,
            "string_length_8": 8,
            "string_length_32": 32
        }
        
        self.requiredVars = [
            "volume_number",
            "time_coverage_start",
            "time_coverage_end",
            "latitude",
            "longitude",
            "altitude",
            "sweep_number",
            "sweep_mode",
            "fixed_angle",
            "sweep_start_ray_index",
            "sweep_end_ray_index",
            "time",
            "range",
            "azimuth",
            "elevation",
            "pulse_width",
            "prt",
            "wavelength",
            "nyquist_velocity"
        ]
        
        self.optionalVarBools = {
            "pulse_boundaries": False
        }
        
        self.radarVarBools = {
            #At least one is required true
            "DBZ": False,
            "VEL": False,
            "WIDTH": False,
            "ZDR": False,
            "PHIDP": False,
            "RHOHV": False,
            "SNRH": False,
            "SNRV": False,
        }
        
        self.variables = {
            #Required variables here
            "volume_number": {
                "type": "i4",
                "fill_value": -9999,
                "dims": (),
                "standard_name": "data_volume_index_number",
                "data": "tbd"
            },
            "time_coverage_start": {
                "type": "|S1",
                "fill_value": b' ',
                "dims": ("string_length_32",),
                "standard_name": "data_volume_start_time_utc",
                "comment": "ray times are relative to start time in secs",
                "data": "tbd"
            },
            "time_coverage_end": {
                "type": "|S1",
                "fill_value": b' ',
                "dims": ("string_length_32",),
                "standard_name": "data_volume_end_time_utc",
                "comment": "ray times are relative to start time in secs",
                "data": "tbd"
            },
            "latitude": {
                "type": "f8",
                "fill_value": -9999.0,
                "dims": (),
                "long_name": "latitude",
                "units": "degrees_north",
                "data": "tbd",
            },
            "longitude": {
                "type": "f8",
                "fill_value": -9999.0,
                "dims": (),
                "long_name": "longitude",
                "units": "degrees_east",
                "data": "tbd",
            },
            "altitude": {
                "type": "f8",
                "fill_value": -9999.0,
                "dims": (),
                "long_name": "altitude",
                "units": "meters",
                "data": 2.5
            },
            "sweep_number": {
                "type": "i4",
                "fill_value": -9999,
                "dims": ("sweep",),
                "long_name": "sweep_index_number_0_based",
                "data": "tbd"
            },
            "sweep_mode": {
                "type": "|S1",
                "fill_value": b' ',
                "dims": ("sweep", "string_length_32"),
                "long_name": "scan_mode_for_sweep",
                "data": "tbd"
            },
            "fixed_angle": {
                "type": "f4",
                "fill_value": -9999.0,
                "dims": ("sweep",),
                "long_name": "ray_target_fixed_angle",
                "units": "tbd",
                "data": "tbd"
            },
            "sweep_start_ray_index": {
                "type": "i4",
                "fill_value": -9999,
                "dims": ("sweep",),
                "long_name": "index_of_first_ray_in_sweep",
                "data": "tbd"
            },
            "sweep_end_ray_index": {
                "type": "i4",
                "fill_value": -9999,
                "dims": ("sweep",),
                "long_name": "index_of_last_ray_in_sweep",
                "data": "tbd"
            },
            "time": {
                "type": "f8",
                "fill_value": -9999.0,
                "dims": ("time",),
                "standard_name": "time",
                "long_name": "time in seconds since volume start",
                "units": "tbd",
                "data": "tbd",
            },
            "range": {
                "type": "f4",
                "fill_value": -9999.0,
                "dims": ("range",),
                "long_name": "Range from instrument to center of gate",
                "units": "meters",
                "spacing_is_constant": "true",
                "meters_to_center_of_first_gate": "tbd",
                "meters_between_gates": "tbd",
                "data": "tbd"
            },
            "azimuth": {
                "type": "f4",
                "fill_value": -9999.0,
                "dims": ("time",),
                "long_name": "ray_azimuth_angle",
                "units": "degrees",
                "data": "tbd"
            },
            "elevation": {
                "type": "f4",
                "fill_value": -9999.0,
                "dims": ("time",),
                "long_name": "ray_elevtion_angle",
                "units": "degrees",
                "data": "tbd",
            },
            "pulse_width": {
                "type": "f4",
                "fill_value": -9999.0,
                "dims": ("time",),
                "long_name": "transmitter_pulse_width",
                "units": "seconds",
                "data": "tbd"
            },
            "prt": {
                "type": "f4",
                "fill_value": -9999.0,
                "dims": ("time",),
                "long_name": "pulse_repetition_time",
                "units": "seconds",
                "data": "tbd"
            },
            "wavelength": {
                "type": "f4",
                "fill_value": -9999.0,
                "dims": ("time",),
                "long_name": "radar_wavelength",
                "units": "meters",
                "data": "tbd"
            },
            "nyquist_velocity": {
                "type": "f4",
                "fill_value": -9999.0,
                "dims": ("time",),
                "long_name": "unambiguous_doppler_velocity",
                "units": "meters per second",
                "data": "tbd"
            },
            
            #Add optional variables here
            "pulse_boundaries": {
                "type": "i4",
                "fill_value": -9999,
                "dims": ("time", "ray_start_end"),
                "long_name": "first_and_last_pulse_indices_in_ray",
                "comment": "First and last pulse index in a ray in corresponding rkc file. Values "
                           "valid after filtering pulses with goodData.csv and badPulseSwaths.csv.",
                "data": "tbd"
            },
            
            #Add radar variables here
            "DBZ": {
                "type": "i2",
                "fill_value": -32768,
                "dims": ("time", "range"),
                "long_name": "reflectivity",
                "standard_name": "equivalent_reflectivity_factor",
                "units": "dBZ",
                "scale_factor": 0.01,
                "add_offset": 0.0,
                "grid_mapping": "grid_mapping",
                "coordinates": "time range",
                "data": "tbd"
            },
            "VEL": {
                "type": "i2",
                "fill_value": -32768,
                "dims": ("time", "range"),
                "long_name": "doppler_velocity",
                "standard_name": "radial_velocity_of_scatterers_away_from_instrument",
                "units": "m/s",
                "scale_factor": 0.01,
                "add_offset": 0.0,
                "grid_mapping": "grid_mapping",
                "coordinates": "time range",
                "data": "tbd"
            },
            "WIDTH": {
                "type": "i2",
                "fill_value": -32768,
                "dims": ("time", "range"),
                "long_name": "spectrum_width",
                "standard_name": "doppler_spectrum_width",
                "units": "m/s",
                "scale_factor": 0.01,
                "add_offset": 0.0,
                "grid_mapping": "grid_mapping",
                "coordinates": "time range",
                "data": "tbd"
            },
            "ZDR": {
                "type": "i2",
                "fill_value": -32768,
                "dims": ("time", "range"),
                "long_name": "differential_reflectivity",
                "standard_name": "log_differential_reflectivity_hv",
                "units": "dB",
                "scale_factor": 0.01,
                "add_offset": 0.0,
                "grid_mapping": "grid_mapping",
                "coordinates": "time range",
                "data": "tbd"
            },
            "RHOHV": {
                "type": "i2",
                "fill_value": -32768,
                "dims": ("time", "range"),
                "long_name": "cross_correlation_ratio",
                "standard_name": "cross_correlation_ratio_hv",
                "units": "unitless",
                "scale_factor": 0.01,
                "add_offset": 0.0,
                "grid_mapping": "grid_mapping",
                "coordinates": "time range",
                "data": "tbd"
            },
            "PHIDP": {
                "type": "i2",
                "fill_value": -32768,
                "dims": ("time", "range"),
                "long_name": "differential_phase",
                "standard_name": "differential_phase_hv",
                "units": "tbd",
                "scale_factor": 0.01,
                "add_offset": 0.0,
                "grid_mapping": "grid_mapping",
                "coordinates": "time range",
                "data": "tbd"
            },
            "SNRH": {
                "type": "i2",
                "fill_value": -32768,
                "dims": ("time", "range"),
                "long_name": "horizontal_channel_signal_to_noise_ratio",
                "standard_name": "signal_to_noise_ratio_h",
                "units": "dB",
                "scale_factor": 0.01,
                "add_offset": 0.0,
                "grid_mapping": "grid_mapping",
                "coordinates": "time range",
                "data": "tbd"
            },
            "SNRV": {
                "type": "i2",
                "fill_value": -32768,
                "dims": ("time", "range"),
                "long_name": "vertical_channel_signal_to_noise_ratio",
                "standard_name": "signal_to_noise_ratio_v",
                "units": "dB",
                "scale_factor": 0.01,
                "add_offset": 0.0,
                "grid_mapping": "grid_mapping",
                "coordinates": "time range",
                "data": "tbd"
            },
        }



# --------------------------------------------------
# Calculate vorticity
# --------------------------------------------------
def calc_vort_radar(radar, vel_dea_str, vortz_str, az_sm_idx, r_sm_idx):
    """
    Calculate inferred vertical vorticity in radar coordinates

    :param radar: pyart radar object
    :param str vel_dea_str: string for dealiased velocity
    :param str vortz_str: string for output vorticity
    :param int smooth_idx: number of azimuth/range indices to smooth vr over
    """

    vortz_full = np.zeros_like(radar.fields[vel_dea_str]['data'])
    for swp in range(radar.nsweeps):
        # extract sweep
        radswp = radar.extract_sweeps([swp])
        # extract fields
        az = radswp.azimuth['data']
        r = radswp.range['data']
        vr = radswp.fields[vel_dea_str]['data']
        # smooth vr
        vr_da = xr.DataArray(vr, dims=["az", "r"])
        vr = vr_da.rolling(az=az_sm_idx,r=r_sm_idx, center=True, min_periods=1).median().values
        # calculate vorticity
        vortz = np.zeros_like(vr)
        vortz[1:,:] = ( (vr[1:,:] - vr[:-1,:]) / (np.deg2rad(az[1:]) - np.deg2rad(az[:-1]))[:,np.newaxis] ) * (1 / r)
        # vortz = np.gradient(vr, az[:,np.newaxis], axis=0) * (1 / r)
        # add to sweep
        vortz_field = {
            'data': vortz,
            'units': '/s',
            'long_name': 'Inferred vertical vorticity from azimuthal shear',
            'standard_name': 'Vertical vorticity',
        }
        radswp.add_field(vortz_str, vortz_field, replace_existing=True)
        # combine sweeps
        if swp == 0:
            azidx = az.size
            vortz_full[:azidx,:] = vortz
        else:
            vortz_full[azidx:azidx+az.size,:] = vortz
            azidx = azidx + az.size
    # add to radar object
    vortz_full_field = {
        'data': vortz_full,
        'units': '/s',
        'long_name': 'Inferred vertical vorticity',
        'standard_name': 'Inferred vertical vorticity',
    }
    radar.add_field(vortz_str, vortz_full_field, replace_existing=True)

    return radar