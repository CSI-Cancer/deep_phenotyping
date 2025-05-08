import numpy as np
import cv2
import os
import h5py

def pdf_to_cdf(pdf):
    """
    Convert a probability density function to a cumulative distribution function.
    """
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    return cdf

def create_lookup(cdf_new, cdf_ref):
    """
    Create a lookup table to map the new image's intensities to match the reference CDF.
    """
    # Ensure cdf_ref is strictly increasing for interpolation
    # Add a small epsilon to cdf_ref to avoid issues with duplicate values
    epsilon = 1e-6
    cdf_ref = np.clip(cdf_ref, epsilon, 1 - epsilon)

    lookup_table = np.interp(cdf_new, cdf_ref, np.arange(len(cdf_ref)))
    return np.round(lookup_table).astype(np.uint16)

def apply_lookup(image_channel, lookup, n_bins, ranges):
    """
    Apply the lookup table to the image channel.

    Parameters:
    - image_channel: The image channel to transform.
    - lookup: The lookup table.
    - n_bins: Number of bins used for histogram.
    - ranges: Range of intensity values.
    """

    # Map image intensities to bin indices
    bin_indices = ((image_channel - ranges[0]) / (ranges[1] - ranges[0])) * (n_bins - 1)
    bin_indices = np.clip(bin_indices.astype(int), 0, n_bins - 1)
    # Apply the lookup table
    mapped_bin_indices = lookup[bin_indices]
    # Map bin indices back to intensity values
    mapped_image = ((mapped_bin_indices / (n_bins - 1)) * (ranges[1] - ranges[0])) + ranges[0]
    return mapped_image.astype(image_channel.dtype)

def load_normalization_config(hdf5_file_path, params):
    """
    Load normalization configuration from an HDF5 file.

    Returns:
    - A dictionary mapping channel names to their normalization method, value, n_bins, and ranges.
    """
    if not os.path.exists(hdf5_file_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_file_path}")

    normalization_config = {}
    with h5py.File(hdf5_file_path, 'r') as h5file:
        for channel in params['channel_names']:
            channel_lower = channel.lower()
            if channel in h5file:
                group = h5file[channel]
            elif channel_lower in h5file:
                group = h5file[channel_lower]
            else:
                print(f"Warning: Channel '{channel}' not found in HDF5 file. Skipping.")
                continue

            # Read the normalization method
            method = group.attrs.get('method', None)
            if isinstance(method, bytes):
                method = method.decode('utf-8')  # Decode bytes to string if necessary
            if method == 'None' or method is None:
                method = None

            # Initialize channel configuration
            channel_config = {
                'method': method,
                'value': None,
                'n_bins': None,
                'ranges': None
            }

            # Read the value and additional parameters
            if method in ['pdf', 'cdf']:
                # Read n_bins
                n_bins = group.attrs.get('n_bins', None)
                if n_bins is None:
                    raise ValueError(f"Missing 'n_bins' for channel '{channel}' in HDF5 file.")
                channel_config['n_bins'] = int(n_bins)

                # Read ranges
                if 'ranges' in group:
                    ranges = group['ranges'][:]
                    if len(ranges) != 2:
                        raise ValueError(f"'ranges' dataset for channel '{channel}' must have 2 elements.")
                    channel_config['ranges'] = ranges
                else:
                    raise ValueError(f"Missing 'ranges' dataset for channel '{channel}' in HDF5 file.")

                # Read value
                if 'value' in group:
                    value = group['value'][:]
                else:
                    print(f"Warning: 'value' dataset not found for channel '{channel}'. Skipping.")
                    continue
                if method == 'pdf':
                    # Convert PDF to CDF
                    value = pdf_to_cdf(value.flatten())
                else:
                    value = value.flatten()
                channel_config['value'] = value
            elif method == 'median':
                # Read value
                if 'value' in group.attrs:
                    value = group.attrs['value']
                elif 'value' in group:
                    value = group['value'][()]
                else:
                    print(f"Warning: 'value' not found for channel '{channel}'. Skipping.")
                    continue
                # Ensure the median value is a scalar
                if isinstance(value, np.ndarray):
                    value = value.item()
                channel_config['value'] = value
            elif method is None:
                channel_config['value'] = None
            else:
                raise ValueError(f"Unknown normalization method '{method}' for channel '{channel}'.")

            normalization_config[channel_lower] = channel_config
    return normalization_config

def create_frame_cdf(frame, channel_configs, channel_indices):
    """
    Compute the CDFs for each channel in the frame based on the mask.

    Parameters:
    - frame: The frame object containing the image and mask.
    - channel_configs: Dictionary containing normalization configurations per channel.
    - channel_indices: Dictionary mapping channel names to their indices in the image array.

    Returns:
    - A dictionary mapping channel names to their CDFs.
    """
    # Process the mask without modifying the original frame.mask
    processed_mask = ((frame.mask > 0).astype(np.uint8)) * 255
    # Ensure mask dimensions match image
    if processed_mask.ndim == 2:
        processed_mask = processed_mask[..., np.newaxis]

    total_pixels = np.sum(processed_mask > 0)
    if total_pixels == 0:
        raise ValueError("Mask has no foreground pixels.")

    cdfs = {}

    for channel_lower, config in channel_configs.items():
        method = config['method']
        if method not in ['pdf', 'cdf']:
            continue  # No need to compute CDF for channels not using pdf/cdf normalization

        idx = channel_indices.get(channel_lower, None)
        if idx is None or idx >= frame.image.shape[2]:
            print(f"Warning: Channel '{channel_lower}' not found in image. Skipping.")
            continue

        n_bins = config['n_bins']
        ranges = config['ranges']

        #convert ranges to list
        ranges = ranges.tolist()

        hist = cv2.calcHist(
            [frame.image],
            [idx],
            processed_mask,
            [n_bins],
            ranges
        ).flatten()

        pdf = hist / total_pixels
        cdf = pdf_to_cdf(pdf)
        cdfs[channel_lower] = cdf

    return cdfs

def apply_normalization(image_channel, frame_cdf, reference_value, method, n_bins=None, ranges=None,
                        current_median=None, mask=None):
    """
    Apply the specified normalization method to the image channel.

    Parameters:
    - image_channel: The image channel to normalize.
    - frame_cdf: The CDF of the image channel.
    - reference_value: The reference data (CDF, median value, etc.).
    - method: The normalization method ('pdf', 'cdf', 'median').
    - n_bins: Number of bins used for histogram (required for 'pdf' and 'cdf' methods).
    - ranges: Range of intensity values (required for 'pdf' and 'cdf' methods).
    - current_median: The current median value of the image channel (if available).
    - mask: The mask array (if needed).

    Returns:
    - Normalized image channel.
    """
    if method in ['pdf', 'cdf']:
        if n_bins is None or ranges is None:
            raise ValueError(f"'n_bins' and 'ranges' must be provided for method '{method}'.")
        ref_cdf = reference_value
        if frame_cdf is None:
            print(f"Warning: Frame CDF is None for method '{method}'. Skipping normalization.")
            return image_channel
        # Validate CDF length
        if n_bins != len(ref_cdf):
            raise ValueError("Number of bins in 'n_bins' is not equal to the number of bins in the reference CDF.")
        # Create lookup table
        lookup = create_lookup(frame_cdf, ref_cdf)
        # Apply the lookup table
        transformed_channel = apply_lookup(image_channel, lookup, n_bins, ranges)
        return transformed_channel
    elif method == 'median':
        median_value = reference_value
        if current_median is None:
            # Compute the current median of the image channel within the mask
            if mask is not None:
                channel_values = image_channel[mask > 0]
                if channel_values.size == 0:
                    print(f"Warning: No valid pixels for median normalization in channel. Skipping.")
                    return image_channel
                current_median = np.median(channel_values)
            else:
                current_median = np.median(image_channel[image_channel > 0])  # Exclude zero values if needed
        # Avoid division by zero
        scaling_factor = median_value / current_median if current_median != 0 else 1
        transformed_channel = image_channel * scaling_factor
        # Clip to the valid range (assuming 16-bit images)
        transformed_channel = np.clip(transformed_channel, 0, 65535)
        return transformed_channel.astype(image_channel.dtype)
    else:
        # If method is None or unrecognized, return the original channel
        return image_channel

def match_image_to_reference(frame, params, hdf5_file_path):
    """
    Match the image channels to the reference distributions based on the normalization configuration.

    Parameters:
    - frame: An object containing `image` (numpy array) and `mask` (numpy array).
    - params: Dictionary containing 'channel_names'.
    - hdf5_file_path: Path to the HDF5 file containing normalization configurations.

    Returns:
    - Transformed image as a numpy array.
    """
    # Load normalization configuration from HDF5 file
    normalization_config = load_normalization_config(hdf5_file_path, params)

    # Map channel names to indices
    channel_names = [x.lower() for x in params['channel_names']]
    channel_indices = {channel: idx for idx, channel in enumerate(channel_names)}

    # Compute CDFs for the frame
    frame_cdfs = create_frame_cdf(frame, normalization_config, channel_indices)

    # Compute current medians
    frame_medians = {}

    for channel_lower, config in normalization_config.items():
        method = config['method']
        if method != 'median':
            continue  # No need to compute medians for other methods

        idx = channel_indices.get(channel_lower, None)
        if idx is None or idx >= frame.image.shape[2]:
            print(f"Warning: Channel '{channel_lower}' not found in image. Skipping.")
            continue

        image_channel = frame.image[:, :, idx]
        channel_values = image_channel[frame.mask > 0]
        if channel_values.size == 0:
            print(f"Warning: No valid pixels for median calculation in channel '{channel_lower}'. Skipping.")
            continue
        frame_medians[channel_lower] = np.median(channel_values)

    # Make a copy of the image to avoid modifying the original
    transformed_image = frame.image.copy()

    # Apply normalization per channel
    for channel_lower, config in normalization_config.items():
        method = config['method']
        reference_value = config['value']
        n_bins = config.get('n_bins', None)
        ranges = config.get('ranges', None)

        if method is None:
            print(f"Channel '{channel_lower}': No normalization applied.")
            continue  # Skip normalization for this channel

        idx = channel_indices.get(channel_lower, None)
        if idx is None or idx >= transformed_image.shape[2]:
            print(f"Warning: Channel '{channel_lower}' not found in image. Skipping.")
            continue

        image_channel = transformed_image[:, :, idx]
        frame_cdf = frame_cdfs.get(channel_lower, None)
        current_median = frame_medians.get(channel_lower, None)


        # Apply the specified normalization method
        transformed_channel = apply_normalization(
            image_channel, frame_cdf, reference_value, method,
            n_bins=n_bins, ranges=ranges,
            current_median=current_median, mask=frame.mask
        )
        transformed_image[:, :, idx] = transformed_channel

    return transformed_image
