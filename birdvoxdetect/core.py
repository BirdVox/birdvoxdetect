import collections
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
        logger_level=20,
        detector_name="pcen_snr"):
        #detector_name="birdvoxdetect_pcen_cnn_adaptive-threshold-T1800"):

    # Check for existence of the input file.
    if not os.path.exists(filepath):
        raise BirdVoxDetectError(
            'File "{}" could not be found.'.format(filepath))

    # Load the file.
    try:
        sound_file = sf.SoundFile(filepath)
    except Exception:
        raise BirdVoxDetectError(
            'Could not open file "{}":\n{}'.format(filepath,
            traceback.format_exc()))

    # Load the detector.
    if detector_name == "pcen_snr":
        detector = "pcen_snr"
    else:
        model_path = os.path.join("models", detector_name + ".h5")
        if not os.path.exists(model_path):
            raise BirdVoxDetectError(
                'Model "{}" could not be found.'.format(detector_name))
        try:
            with warnings.catch_warnings():
                # Suppress TF and Keras warnings when importing
                warnings.simplefilter("ignore")
                import keras
                detector = keras.models.load_model(model_path)
        except Exception:
            raise BirdVoxDetectError(
                'Could not open model "{}":\n{}'.format(filepath,
                traceback.format_exc()))

    # Define chunk size.
    has_context = len(detector_name)>6 and (detector_name[-6:-4] == "-T")
    if has_context:
        percentiles = [0.1, 1, 10, 25, 50, 75, 90, 99, 99.9]
        queue_length = 4
        chunk_duration = int(detector_name[-4:]) / queue_length
    else:
        chunk_duration = 450
        queue_length = 1

    # Define number of chunks.
    sr = sound_file.samplerate
    chunk_length = int(chunk_duration * sr)
    full_length = len(sound_file)
    n_chunks = max(1, int(np.ceil(full_length) / chunk_length))

    # Pre-load double-ended queue.
    deque = collections.deque()
    for chunk_id in range(min(n_chunks, queue_length)):
        chunk_start = chunk_id * chunk_length
        sound_file.seek(chunk_start)
        chunk_audio = sound_file.read(chunk_length)
        chunk_pcen = compute_pcen(chunk_audio, sr)
        deque.append(chunk_pcen)

    # Compute context.
    if has_context:
        concat_deque = np.concatenate(deque, axis=1)
        deque_context = np.percentile(percentiles, axis=1)

    # Compute likelihood on queue chunks.
    chunk_likelihoods = []
    for chunk_id in range(min(queue_length, n_chunks-1)):
        chunk_pcen = deque[chunk_id]
        if has_context:
            chunk_likelihood = predict_with_context(
                chunk_pcen, frame_rate, deque_context, detector)
        else:
            chunk_likelihood = predict(chunk_pcen, frame_rate, detector)
        chunk_likelihoods.append(chunk_likelihood)

    # Loop over chunks.
    for chunk_id in range(queue_length, n_chunks-1):
        chunk_start = chunk_id * chunk_length
        sound_file.seek(chunk_start)
        chunk_audio = sound_file.read(chunk_length)
        deque.popleft()
        chunk_pcen = compute_pcen(chunk_audio, sr)
        deque.append(chunk_pcen)
        concat_deque = np.concatenate(deque, axis=1, out=concat_deque)
        deque_context = np.percentile(percentiles, axis=1, out=deque_context)
        if has_context:
            chunk_likelihood = predict_with_context(
                chunk_pcen, frame_rate, deque_context, detector)
        else:
            chunk_likelihood = predict(chunk_pcen, frame_rate, detector)
        chunk_likelihoods.append(chunk_likelihood)

    # Last chunk.
    chunk_start = (n_chunks-1) * chunk_length
    sound_file.seek(chunk_start)
    chunk_audio = sound_file.read(full_length - chunk_start)
    chunk_pcen = compute_pcen(chunk_audio, sr)
    if has_context:
        if n_chunks == 1:
            deque_context = np.percentile(chunk_pcen, percentiles, axis=1)
        chunk_likelihood = predict_with_context(
            chunk_pcen, deque_context, frame_rate, detector)
    else:
        chunk_likelihood = predict(
            chunk_pcen, frame_rate, detector)
    chunk_likelihoods.append(chunk_likelihood)

    # Concatenate predictions.
    likelihood = np.concatenate(chunk_likelihoods)

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


def compute_pcen(audio, sr):
    # Load settings.
    pcen_settings = get_pcen_settings()

    # Resample to 22,050 kHz
    if not sr == pcen_settings["sr"]:
        audio = librosa.resample(audio, sr, pcen_settings["sr"])
        sr = pcen_settings["sr"]

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

    # Return.
    return pcen


def predict(pcen, frame_rate, detector):
    pcen_settings = get_pcen_settings()

    # PCEN-SNR
    if detector == "pcen_snr":
        pcen_snr = np.max(pcen, axis=0) - np.min(pcen, axis=0)
        pcen_likelihood = pcen_snr / (0.001 + pcen_snr)
        median_likelihood = scipy.signal.medfilt(
            pcen_likelihood, kernel_size=127)
        sr = pcen_settings["sample_rate"]
        hop_length = pcen_settings["hop_length"]
        audio_duration = pcen.shape[0]*hop_length/sr
        likelihood_x = np.arange(
            0.0,
            audio_duration/pcen_settings["hop_length"],
            sr/(pcen_settings["hop_length"]*frame_rate))[:-1].astype('int')
        y = median_likelihood[likelihood_x]
        return y

    # PCEN-CNN. (no context adaptation)
    else:
        # Compute number of hops.
        clip_length = 104
        hop_length = 34
        n_freqs, n_times = pcen.shape
        n_hops = 1 + int((n_times - clip_length) / hop_length)
        itemsize = pcen.itemsize

        # Stride and tile.
        X_shape = (n_hops, clip_length, n_freqs)
        X_stride = (itemsize*n_freqs*hop_length, itemsize*n_freqs, itemsize)
        X_pcen = np.lib.stride_tricks.as_strided(
            np.ravel(np.copy(pcen).T),
            shape=X_shape,
            strides=X_stride,
            writeable=False)
        X_pcen = np.transpose(X_pcen, (0, 2, 1))[:, :, :, np.newaxis]

        # Predict.
        y = detector.predict({"spec_input": X_pcen})

        # Return likelihood.
        return (1 - y)


def predict_with_context(pcen, context, frame_rate, detector):
    # Compute number of hops.
    clip_length = 104
    hop_length = 34
    n_freqs, n_times = pcen.shape
    n_hops = 1 + int((n_times - clip_length) / hop_length)
    itemsize = pcen.itemsize

    # Stride and tile.
    X_shape = (n_hops, clip_length, n_freqs)
    X_stride = (itemsize*n_freqs*hop_length, itemsize*n_freqs, itemsize)
    X_pcen = np.lib.stride_tricks.as_strided(
        np.ravel(np.copy(pcen).T),
        shape=X_shape,
        strides=X_stride,
        writeable=False)
    X_pcen = np.transpose(X_pcen, (0, 2, 1))[:, :, :, np.newaxis]
    X_bg = np.tile(pcen_percentiles, (n_hops, 1, 1))

    # Predict.
    y = detector.predict({"spec_input": X_pcen, "bg_input": X_bg})

    # Return likelihood.
    return (1 - y)


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
