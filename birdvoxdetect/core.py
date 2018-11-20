import os
import soundfile as sf

TARGET_SR = 22050

def _center_audio(audio, frame_len):
    """Center audio so that first sample will occur in the middle of the first frame"""
    return np.pad(audio, (int(frame_len / 2.0), 0), mode='constant', constant_values=0)


def process_file(filepath,
        output_dir=None,
        export_clips=False,
        threshold=50.0,
        suffix="",
        frame_rate=20.0,
        clip_duration=1.0,
        verbose=True):
    
    # Check for existence of the file.
    if not os.path.exists(filepath):
        raise BirdVoxDetectError('File "{}" could not be found.'.format(filepath))
        
    # Try loading the file as NumPy array.
    try:
        audio, sr = sf.read(filepath)
    except Exception:
        raise OpenL3Error('Could not open file "{}":\n{}'.format(filepath, traceback.format_exc()))
        
    # Define output path.
    output_path = get_output_path(filepath, suffix + ".csv", output_dir=output_dir)

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
