import h5py
import librosa
import numpy as np
import os
import pandas as pd
import scipy.signal
import soundfile as sf
import traceback


from birdvoxdetect.birdvoxdetect_exceptions import BirdVoxDetectError


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

    # Check for existence of the input file.
    if not os.path.exists(filepath):
        raise BirdVoxDetectError(
            'File "{}" could not be found.'.format(filepath))

    # Try loading the file as NumPy array.
    try:
        audio, sr = sf.read(filepath)
    except Exception:
        raise BirdVoxDetectError(
            'Could not open file "{}":\n{}'.format(filepath,
            traceback.format_exc()))

    # Compute likelihood curve.
    likelihood = get_likelihood(
        audio, pcen_settings["sr"], frame_rate=frame_rate)

    # Find peaks.
    peak_locs, _ = scipy.signal.find_peaks(likelihood)

    # Threshold peaks.
    th_peak_locs = peak_locs[peak_locs>threshold/100]
    th_peak_likelihoods = likelihood[th_peak_locs]
    th_peak_timestamps = th_peak_locs / frame_rate

    # Create output_dir if necessary.
    if output_dir is not None:
        try:
            os.makedirs(output_dir)
        except OSError:
            pass

    # Append underscore to suffix if it is not empty.
    if len(suffix) > 0 and not suffix[-1] == "_":
        suffix = suffix + "_"

    # Export timestamps.
    timestamps_path = get_output_path(
        filepath, suffix + "timestamps.csv", output_dir=output_dir)
    df_matrix = np.stack(
        (th_peak_timestamps, th_peak_likelihoods), axis=1)
    df = pd.DataFrame(df_matrix, columns=["Time (s)", "Likelihood (%)"])
    df.to_csv(timestamps_path)

    # Export likelihood curve.
    if export_likelihood:
        likelihood_path = get_output_path(
            filepath, suffix + "likelihood.hdf5", output_dir=output_dir)

        with h5py.File(likelihood_path, "w") as f:
            f.create_dataset('likelihood', data=likelihood)

    # Export clips.
    if export_clips:
        clips_dir = get_output_path(
            filepath, suffix + "clips", output_dir=output_dir)
        try:
            os.makedirs(clips_dir)
        except OSError:
            pass

        for t in th_peak_timestamps:
            start = int(sr*np.round(t-0.5*clip_duration))
            stop = int(sr*np.round(t+0.5*clip_duration))
            audio_clip = audio[start:stop]
            clip_path = get_output_path(
                filepath,
                suffix + "{:05.2f}".format(t).replace(".", "-") + ".wav",
                output_dir = clips_dir)
            librosa.output.write_wav(clip_path, audio_clip, sr)


def get_likelihood(audio, sr, frame_rate, detector="pcen_snr"):
    # Load settings.
    pcen_settings = get_pcen_settings()

    # Check audio array dimension
    if audio.ndim > 2:
        raise BirdVoxDetectError('Audio array can only be be 1D or 2D')
    elif audio.ndim == 2:
        # Downmix if multichannel
        audio = np.mean(audio, axis=1)

    # Resample to 22,050 kHz
    if not sr == pcen_settings["sr"]:
        audio = librosa.resample(audio, sr, pcen_settings["sr"])

    # Pad.
    padding_length = int(np.round(0.5 * sr / frame_rate))
    audio = pad_audio(audio, padding_length)

    # Compute Short-Term Fourier Transform (STFT).
    stft = librosa.stft(
        audio,
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

    # PCEN-SNR detector.
    if detector == "pcen_snr":
        pcen_snr = np.max(pcen, axis=0) - np.min(pcen, axis=0)
        pcen_likelihood = pcen_snr / (0.001 + pcen_snr)
        median_likelihood = scipy.signal.medfilt(pcen_likelihood,
            kernel_size=127)
        audio_duration = audio.shape[0]
        likelihood_x = np.arange(
            0.0,
            audio_duration/pcen_settings["hop_length"],
            sr/(pcen_settings["hop_length"]*frame_rate)).astype('int')
        likelihood_y = median_likelihood[likelihood_x]

    # Deep learning detector with PCEN input and context adaptation.
    elif detector == "pcen_cnn_adaptive-threshold-T1800":
        pass

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
        "hop_length": 32,
        "n_fft": 1024,
        "n_mels": 128,
        "pcen_delta": 10.0,
        "pcen_time_constant": 0.06,
        "pcen_norm_exponent": 0.8,
        "pcen_power": 0.25,
        "sr": 22050.0,
        "win_length": 256,
        "window": "hann"}
    return pcen_settings
