import librosa
import numpy as np
import os
import scipy.signal
import soundfile as sf


def pad_audio(audio, padding_length):
    """Pad audio so that first sample will occur in the middle of the first frame"""
    return np.pad(audio, (int(padding_length), 0),
                  mode='constant', constant_values=0)


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
        
    # Resample to 22,050 kHz
    if not sr == pcen_settings["sr"]:
        audio = librosa.resample(audio, sr, pcen_settings["sr"])
        
    # Pad.
    padding_length = int(np.round(0.5 * sr / frame_rate))
    audio = pad_audio(audio, padding_length)
        
    # Load settings.
    pcen_settings = get_pcen_settings()
    
    # Compute Short-Term Fourier Transform (STFT).
    stft = librosa.stft(
        chunk_waveform,
        n_fft=pcen_settings["n_fft"],
        win_length=pcen_settings["win_length"],
        hop_length=pcen_settings["hop_length"],
        window=pcen_settings["window"])

    # Compute squared magnitude coefficients.
    abs2_stft = (stft.real*stft.real) + (stft.imag*stft.imag)

    # Gather frequency bins according to the Mel scale.
    melspec = librosa.feature.melspectrogram(
        y=None,
        S=abs2_stft,
        sr=pcen_settings["sr"],
        n_fft=pcen_settings["n_fft"],
        n_mels=pcen_settings["n_mels"],
        htk=True,
        fmin=pcen_settings["fmin"],
        fmax=pcen_settings["fmax"])

    # Compute PCEN.
    pcen = librosa.pcen(melspec,
        sr=pcen_settings["sr"],
        hop_length=pcen_settings["hop_length"],
        gain=pcen_settings["pcen_norm_exponent"],
        bias=2,
        power=pcen_settings["pcen_power"],
        time_constant=pcen_settings["pcen_time_constant"])

    # Convert to single floating-point precision.
    pcen = pcen.astype('float32')

    # Compute likelihood curve.
    pcen_snr = np.max(pcen, axis=0) - np.min(pcen, axis=0)
    pcen_likelihood = pcen_snr / (1.0 + pcen_snr)
    median_likelihood = scipy.signal.medfilt(pcen_likelihood,
        kernel_size=128)
    fractional_subsampling =\
        pcen_settings["sr"] / (pcen_settings["hop_length"]*frame_rate)
    audio_duration = len(audio) / pcen_settings["sr"]
    likelihood_x = np.arange(0.0, audio_duration, 1.0/frame_rate)
    likelihood_y = median_likelihood[likelihood_x]
    
    # Return.
    return likelihood_y
    

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


def get_pcen_settings():
    pcen_settings = {
        "fmin": 2000.0,
        "fmax": 11025.0,
        "hop_length": 32.0,
        "n_fft": 1024,
        "n_mels": 128,
        "pcen_delta": 10.0,
        "pcen_time_constant": 0.06,
        "pcen_norm_exponent": 0.8,
        "pcen_power": 0.25,
        "sr": 22050.0,
        "win_length": 256.0,
        "window": "hann"}
    return pcen_settings
