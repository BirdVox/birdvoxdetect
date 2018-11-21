import os
import soundfile as sf

TARGET_SR = 22050

def _center_audio(audio, frame_len):
    """Center audio so that first sample will occur in the middle of the first frame"""
    return np.pad(audio, (int(frame_len / 2.0), 0), mode='constant', constant_values=0)


def process_file(filepath,
        output_dir=None,
        export_clips=False,
        export_likelihood=False,
        threshold=50.0,
        suffix="",
        frame_rate=20.0,
        clip_duration=1.0,
        logger_level=20):
    
    # Check for existence of the file.
    if not os.path.exists(filepath):
        raise BirdVoxDetectError('File "{}" could not be found.'.format(filepath))
        
    # Try loading the file as NumPy array.
    try:
        audio, sr = sf.read(filepath)
    except Exception:
        raise OpenL3Error('Could not open file "{}":\n{}'.format(filepath, traceback.format_exc()))
        
    # Compute likelihood curve.
    likelihood = get_likelihood(audio, sr, frame_rate=frame_rate)
    
    # Find peaks.
    
    # Export timestamps.
    timestamps_path = get_output_path(
        filepath, suffix + "_timestamps.csv", output_dir=output_dir)
    
    # Export likelihood curve.
    if export_likelihood:
        likelihood_path = get_output_path(
            filepath, suffix + "_likelihood.hdf5", output_dir=output_dir)
        
    # Export clips.
    if export_clips:
        clips_dir = get_output_path(
            filepath, suffix + "_clips", output_dir=output_dir)


def get_likelihood(audio, sr, frame_rate):
    # Check audio array dimension
    if audio.ndim > 2:
        raise OpenL3Error('Audio array can only be be 1D or 2D')
    elif audio.ndim == 2:
        # Downmix if multichannel
        audio = np.mean(audio, axis=1)
        
        
    raise NotImplementedError()
        
        
        

def get_output_path(filepath, suffix, output_dir=None):
    """
    Parameters
    ----------
    filepath : str
        Path to audio file to be processed
    suffix : str
        String to append to filename (including extension)
    output_dir : str or None
        Path to directory where file will be saved.
        If None, will use directory of given filepath.
    Returns
    -------
    output_path : str
        Path to output file
    """
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    if not output_dir:
        output_dir = os.path.dirname(filepath)

    if suffix[0] != '.':
        output_filename = "{}_{}".format(base_filename, suffix)
    else:
        output_filename = base_filename + suffix

    return os.path.join(output_dir, output_filename)
