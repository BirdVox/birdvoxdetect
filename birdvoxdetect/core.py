import collections
import h5py
import librosa
import logging
import numpy as np
import os
import pandas as pd
import scipy.signal
import soundfile as sf
import traceback
import warnings


from birdvoxdetect.birdvoxdetect_exceptions import BirdVoxDetectError


def process_file(
        filepath,
        output_dir=None,
        export_clips=False,
        export_likelihood=False,
        threshold=50.0,
        suffix="",
        frame_rate=20.0,
        clip_duration=1.0,
        logger_level=logging.INFO,
        detector_name="pcen_snr"):
    # detector_name="birdvoxdetect_pcen_cnn_adaptive-threshold-T1800"):
    # Set logger level.
    logging.getLogger().setLevel(logger_level)

    # Print new line and file name.
    logging.info("-" * 75)
    logging.info("Loading file: {}".format(filepath))

    # Check for existence of the input file.
    if not os.path.exists(filepath):
        raise BirdVoxDetectError(
            'File "{}" could not be found.'.format(filepath))

    # Load the file.
    try:
        sound_file = sf.SoundFile(filepath)
    except Exception:
        exc_str = 'Could not open file "{}":\n{}'
        exc_formatted_str = exc_str.format(filepath, traceback.format_exc())
        raise BirdVoxDetectError(exc_formatted_str)

    # Print model.
    logging.info("Loading model: {}".format(detector_name))

    # Load the detector.
    if detector_name == "pcen_snr":
        detector = "pcen_snr"
    else:
        model_path = get_model_path(detector_name)
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
            exc_str = 'Could not open model "{}":\n{}'
            formatted_trace = traceback.format_exc()
            exc_formatted_str = exc_str.format(filepath, formatted_trace)
            raise BirdVoxDetectError(exc_formatted_str)

    # Define chunk size.
    has_context = (len(detector_name) > 6) and (detector_name[-6:-4] == "-T")
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

    # Append underscore to suffix if it is not empty.
    if len(suffix) > 0 and not suffix[-1] == "_":
        suffix = suffix + "_"

    # Create output_dir if necessary.
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Initialize file of timestamps.
    if threshold is not None:
        timestamps_path = get_output_path(
            filepath, suffix + "timestamps.csv", output_dir=output_dir)
        event_times = []
        event_likelihoods = []
        df = pd.DataFrame({
            "Time (s)": event_times,
            "Likelihood (%)": event_likelihoods
        })
        df.to_csv(timestamps_path, index=False)

    # Create directory of output clips.
    if export_clips:
        clips_dir = get_output_path(
            filepath, suffix + "clips", output_dir=output_dir)
        if not os.path.exists(clips_dir):
            os.makedirs(clips_dir)

    # Append likelihood to list of per-chunk likelihood.
    if export_likelihood:
        chunk_likelihoods = []

    # Print chunk duration.
    logging.info("Chunk duration: {} seconds".format(chunk_duration))

    # Pre-load double-ended queue.
    deque = collections.deque()
    for chunk_id in range(min(n_chunks-1, queue_length)):
        # Read audio chunk.
        chunk_start = chunk_id * chunk_length
        sound_file.seek(chunk_start)
        chunk_audio = sound_file.read(chunk_length)

        # Compute PCEN.
        chunk_pcen = compute_pcen(chunk_audio, sr)
        deque.append(chunk_pcen)

    # Compute context.
    if has_context:
        concat_deque = np.concatenate(deque, axis=1)
        deque_context = np.percentile(concat_deque, percentiles, axis=1)

    # Compute likelihood on queue chunks.
    for chunk_id in range(min(queue_length, n_chunks-1)):
        # Print chunk ID and number of chunks.
        logging.info("Chunk ID: {}/{}".format(1+chunk_id, n_chunks))

        # Predict.
        chunk_pcen = deque[chunk_id]
        if has_context:
            chunk_likelihood = predict_with_context(
                chunk_pcen, deque_context, frame_rate, detector,
                logger_level)
        else:
            chunk_likelihood = predict(
                chunk_pcen, frame_rate, detector, logger_level)
        chunk_likelihood = np.squeeze(chunk_likelihood)

        # If continuous likelihood is required, store it in memory.
        if export_likelihood:
            chunk_likelihoods.append(chunk_likelihood)

        # If thresholding is not required, jump to next chunk.
        if threshold is None:
            continue

        # Find peaks.
        peak_locs, _ = scipy.signal.find_peaks(chunk_likelihood)
        peak_vals = chunk_likelihood[peak_locs]

        # Threshold peaks.
        th_peak_locs = peak_locs[peak_vals > (threshold/100)]
        th_peak_likelihoods = chunk_likelihood[th_peak_locs]
        chunk_offset = chunk_duration * chunk_id
        th_peak_timestamps = chunk_offset + th_peak_locs/frame_rate
        n_peaks = len(th_peak_timestamps)
        logging.info("Number of timestamps: {}".format(n_peaks))

        # Export timestamps.
        event_times = event_times + th_peak_timestamps
        event_likelihoods = event_likelihoods + th_peak_likelihoods
        df = pd.DataFrame({
            "Time (s)": event_times,
            "Likelihood (%)": event_likelihoods
        })
        df.to_csv(timestamps_path, index=False)

        for t in th_peak_timestamps:
            clip_start = max(0, int(sr*np.round(t-0.5*clip_duration)))
            clip_stop = min(
                len(sound_file), int(sr*np.round(t+0.5*clip_duration)))
            sound_file.seek(clip_start)
            audio_clip = sound_file.read(clip_stop-clip_start)
            clip_path = get_output_path(
                filepath,
                suffix + "{:05.2f}".format(t).replace(".", "-") + ".wav",
                output_dir=clips_dir)
            librosa.output.write_wav(clip_path, audio_clip, sr)

    # Loop over chunks.
    for chunk_id in range(queue_length, n_chunks-1):
        # Print chunk ID and number of chunks.
        logging.info("Chunk ID: {}/{}".format(1+chunk_id, n_chunks))

        # Read chunk.
        chunk_start = chunk_id * chunk_length
        sound_file.seek(chunk_start)
        chunk_audio = sound_file.read(chunk_length)

        # Compute PCEN.
        deque.popleft()
        chunk_pcen = compute_pcen(chunk_audio, sr)
        deque.append(chunk_pcen)

        # Compute percentiles
        concat_deque = np.concatenate(deque, axis=1, out=concat_deque)
        deque_context = np.percentile(
            concat_deque, percentiles, axis=1, out=deque_context)

        # Predict.
        if has_context:
            chunk_likelihood = predict_with_context(
                chunk_pcen, deque_context, frame_rate, detector)
        else:
            chunk_likelihood = predict(chunk_pcen, frame_rate, detector)
        chunk_likelihood = np.squeeze(chunk_likelihood)

        # If continuous likelihood is required, store it in memory.
        if export_likelihood:
            chunk_likelihoods.append(chunk_likelihood)

        # If thresholding is not required, jump to next chunk.
        if threshold is None:
            continue

        # Find peaks.
        peak_locs, _ = scipy.signal.find_peaks(chunk_likelihood)
        peak_vals = chunk_likelihood[peak_locs]

        # Threshold peaks.
        th_peak_locs = peak_locs[peak_vals > (threshold/100)]
        th_peak_likelihoods = chunk_likelihood[th_peak_locs]
        chunk_offset = chunk_duration * chunk_id
        th_peak_timestamps = chunk_offset + th_peak_locs/frame_rate
        n_peaks = len(th_peak_timestamps)
        logging.info("Number of timestamps: {}".format(n_peaks))

        # Export timestamps.
        event_times = event_times + th_peak_timestamps
        event_likelihoods = event_likelihoods + th_peak_likelihoods
        df = pd.DataFrame({
            "Time (s)": event_times,
            "Likelihood (%)": event_likelihoods
        })
        df.to_csv(timestamps_path, index=False)

        # Export clips.
        for t in th_peak_timestamps:
            clip_start = max(0, int(sr*np.round(t-0.5*clip_duration)))
            clip_stop = min(
                len(sound_file), int(sr*np.round(t+0.5*clip_duration)))
            sound_file.seek(clip_start)
            audio_clip = sound_file.read(clip_stop-clip_start)
            clip_path = get_output_path(
                filepath,
                suffix + "{:05.2f}".format(t).replace(".", "-") + ".wav",
                output_dir=clips_dir)
            librosa.output.write_wav(clip_path, audio_clip, sr)

    # Last chunk.
    # Print chunk ID and number of chunks.
    logging.info("Chunk ID: {}/{}".format(1+n_chunks, n_chunks))
    chunk_start = (n_chunks-1) * chunk_length
    sound_file.seek(chunk_start)
    chunk_audio = sound_file.read(full_length - chunk_start)
    chunk_pcen = compute_pcen(chunk_audio, sr)
    if has_context:
        # If the queue is empty, compute percentiles on the fly.
        if n_chunks == 1:
            deque_context = np.percentile(chunk_pcen, percentiles, axis=1)

        # Predict.
        chunk_likelihood = predict_with_context(
            chunk_pcen, deque_context, frame_rate, detector, logger_level)
    else:
        # Predict.
        chunk_likelihood = predict(
            chunk_pcen, frame_rate, detector, logger_level)
    chunk_likelihood = np.squeeze(chunk_likelihood)

    # Threshold last chunk if required.
    if threshold is not None:

        # Find peaks.
        peak_locs, _ = scipy.signal.find_peaks(chunk_likelihood)
        peak_vals = chunk_likelihood[peak_locs]

        # Threshold peaks.
        th_peak_locs = peak_locs[peak_vals > (threshold/100)]
        th_peak_likelihoods = chunk_likelihood[th_peak_locs]
        chunk_offset = chunk_duration * (n_chunks-1)
        th_peak_timestamps = chunk_offset + th_peak_locs/frame_rate
        n_peaks = len(th_peak_timestamps)
        logging.info("Number of timestamps: {}".format(n_peaks))

        # Export timestamps.
        event_times = event_times + th_peak_timestamps
        event_likelihoods = event_likelihoods + th_peak_likelihoods
        df = pd.DataFrame({
            "Time (s)": event_times,
            "Likelihood (%)": event_likelihoods
        })
        df.to_csv(timestamps_path, index=False)

        # Export clips.
        for t in th_peak_timestamps:
            clip_start = max(0, int(sr*np.round(t-0.5*clip_duration)))
            clip_stop = min(
                len(sound_file), int(sr*np.round(t+0.5*clip_duration)))
            sound_file.seek(clip_start)
            audio_clip = sound_file.read(clip_stop-clip_start)
            clip_path = get_output_path(
                filepath,
                suffix + "{:05.2f}".format(t).replace(".", "-") + ".wav",
                output_dir=clips_dir)
            librosa.output.write_wav(clip_path, audio_clip, sr)

    # Export likelihood curve.
    if export_likelihood:

        # Define output path for likelihood.
        likelihood_path = get_output_path(
            filepath, suffix + "likelihood.hdf5", output_dir=output_dir)

        # Concatenate likelihood curves across chunks.
        chunk_likelihoods.append(chunk_likelihood)
        likelihood = np.squeeze(np.concatenate(chunk_likelihoods))
        with h5py.File(likelihood_path, "w") as f:
            f.create_dataset('likelihood', data=likelihood)

    # Print final messages.
    logging.info("Done with file: {}.".format(filepath))
    if threshold is not None:
        timestamp_str = "Timestamps are available at: {}"
        logging.info(timestamp_str.format(timestamps_path))
    if export_clips:
        logging.info("Clips are available at: {}".format(clips_path))
    if export_likelihood:
        logging.info("Likelihood is available at: {}".format(clips_path))



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
    pcen = librosa.pcen(
        melspec,
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


def predict(pcen, frame_rate, detector, logger_level):
    pcen_settings = get_pcen_settings()

    # PCEN-SNR
    if detector == "pcen_snr":
        pcen_snr = np.max(pcen, axis=0) - np.min(pcen, axis=0)
        pcen_likelihood = pcen_snr / (0.001 + pcen_snr)
        median_likelihood = scipy.signal.medfilt(
            pcen_likelihood, kernel_size=127)
        sr = pcen_settings["sr"]
        hop_length = pcen_settings["hop_length"]
        audio_duration = pcen.shape[1]*hop_length/sr
        likelihood_x = np.arange(
            0.0,
            audio_duration*sr/hop_length,
            sr/(hop_length*frame_rate))[:-1].astype('int')
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
        verbose = True
        y = detector.predict(X_pcen, verbose=verbose)

        # Return likelihood.
        return np.maximum(0, 1 - 2*y)


def predict_with_context(pcen, context, frame_rate, detector, logger_level):
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
    X_bg = np.tile(context.T, (n_hops, 1, 1))

    # Predict.
    verbose = True
    y = detector.predict(
        {"spec_input": X_pcen, "bg_input": X_bg},
        verbose=verbose)

    # Return likelihood.
    return np.maximum(0, 1 - 2*y)


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


def get_model_path(model_name):
    return os.path.join(
        os.path.dirname(__file__), "models", model_name + '.h5')
