import collections
import datetime
import h5py
import joblib
import librosa
import logging
import numpy as np
import os
import pandas as pd
import scipy
import scipy.signal
import sklearn
import soundfile as sf
import tensorflow as tf
import traceback
import warnings


import birdvoxdetect
from birdvoxdetect.birdvoxdetect_exceptions import BirdVoxDetectError


def process_file(
        filepath,
        output_dir=None,
        export_clips=False,
        export_confidence=False,
        threshold=30.0,
        suffix="",
        clip_duration=1.0,
        logger_level=logging.INFO,
        detector_name="birdvoxdetect_pcen_cnn_adaptive-threshold-T1800",
        custom_objects=None):

    # Set logger level.
    logging.getLogger().setLevel(logger_level)

    # Print new line and file name.
    logging.info("-" * 72)
    modules = [
        birdvoxdetect, h5py, joblib, librosa, logging,
        np, pd, tf, scipy, sf, sklearn]
    for module in modules:
        logging.info(module.__name__.ljust(15) + " v" + module.__version__)
    logging.info("")
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

    # Load the detector of sensor faults.
    sensorfault_model_path = get_model_path('birdvoxactivate.pkl')
    if not os.path.exists(sensorfault_model_path):
        raise BirdVoxDetectError(
            'Model "{}" could not be found.'.format(detector_name))
    sensorfault_model = joblib.load(sensorfault_model_path)

    # Load the detector of flight calls.
    if detector_name == "pcen_snr":
        detector = "pcen_snr"
    else:
        model_path = get_model_path(detector_name + '.h5')
        if not os.path.exists(model_path):
            raise BirdVoxDetectError(
                'Model "{}" could not be found.'.format(detector_name))
        try:
            with warnings.catch_warnings():
                # Suppress TF and Keras warnings when importing
                warnings.simplefilter("ignore")
                from tensorflow import keras
                detector = keras.models.load_model(
                    model_path, custom_objects=custom_objects)
        except Exception:
            exc_str = 'Could not open model "{}":\n{}'
            formatted_trace = traceback.format_exc()
            exc_formatted_str = exc_str.format(model_path, formatted_trace)
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
        event_confidences = []
        df_columns = ["Time (s)", "Confidence (%)"]
        df = pd.DataFrame({
            "Time (s)": event_times,
            "Confidence (%)": event_confidences
        })
        df.to_csv(
            timestamps_path,
            columns=df_columns, float_format='%8.2f', index=True)

    # Create directory of output clips.
    if export_clips:
        clips_dir = get_output_path(
            filepath, suffix + "clips", output_dir=output_dir)
        if not os.path.exists(clips_dir):
            os.makedirs(clips_dir)

    # Append confidence to list of per-chunk confidences.
    if export_confidence:
        chunk_confidences = []

    # Print chunk duration.
    logging.info("Chunk duration: {} seconds".format(chunk_duration))

    # Define padding. Set to one second, i.e. 750 hops @ 24 kHz.
    # Any value above clip duration (150 ms) would work.
    pcen_settings = get_pcen_settings()
    chunk_padding = pcen_settings["hop_length"] *\
        int(pcen_settings["sr"] / pcen_settings["hop_length"])

    # Pre-load double-ended queue.
    deque = collections.deque()
    for chunk_id in range(min(n_chunks-1, queue_length)):
        # Read audio chunk.
        chunk_start = chunk_id * chunk_length
        sound_file.seek(chunk_start)
        chunk_audio = sound_file.read(chunk_length+chunk_padding)

        # Compute PCEN.
        chunk_pcen = compute_pcen(chunk_audio, sr)
        deque.append(chunk_pcen)

    # Compute context.
    if has_context and (n_chunks>1):
        concat_deque = np.concatenate(deque, axis=1)
        deque_context = np.percentile(
            concat_deque, percentiles, axis=1, overwrite_input=True)

        # Compute sensor fault features.
        # Median is 4th order statistic. Restrict to lowest 120 mel-freq bins
        context_median = deque_context[4, :120]
        context_median_medfilt = scipy.signal.medfilt(
            context_median, kernel_size=(13,))
        sensorfault_features = context_median_medfilt[::12].reshape(1, -1)

        # Compute probability of sensor fault.
        sensor_fault_probability = sensorfault_model.predict(
            sensorfault_features)

        # If probability of sensor fault is above 50%, exclude start of recording
        if sensor_fault_probability > 0.5:
            logging.info("Probability of sensor fault: {:5.2f}%".format(
                100*sensor_fault_probability))
            chunk_id_start = min(n_chunks-1, queue_length)
            context_duration = chunk_duration
            context_duration_str = str(datetime.timedelta(
                seconds=context_duration))
            logging.info(
                "Ignoring segment between 00:00:00 and " +\
                context_duration_str + " (" + chunk_id_start + " chunks)")
        else:
            chunk_id_start = 0

    # Define frame rate.
    frame_rate =\
        pcen_settings["sr"] /\
        (pcen_settings["hop_length"] * pcen_settings["stride_length"])

    # Compute confidence on queue chunks.
    # NB: the following loop is skipped if there is a single chunk.
    for chunk_id in range(chunk_id_start, min(queue_length, n_chunks-1)):
        # Print chunk ID and number of chunks.
        logging.info("Chunk ID: {}/{}".format(
            str(1+chunk_id).zfill(len(str(n_chunks))), n_chunks))

        # Predict.
        chunk_pcen = deque[chunk_id]
        if has_context:
            chunk_confidence = predict_with_context(
                chunk_pcen, deque_context, detector, logger_level,
                padding=chunk_padding)
        else:
            chunk_confidence = predict(
                chunk_pcen, detector, logger_level,
                padding=chunk_padding)

        # Remove trailing singleton dimension
        chunk_confidence = np.squeeze(chunk_confidence)

        # Map confidence to 0-100 range.
        chunk_confidence = map_confidence(chunk_confidence, detector_name)

        # If continuous confidence is required, store it in memory.
        if export_confidence:
            chunk_confidences.append(chunk_confidence)

        # If thresholding is not required, jump to next chunk.
        if threshold is None:
            continue

        # Find peaks.
        peak_locs, _ = scipy.signal.find_peaks(chunk_confidence)
        peak_vals = chunk_confidence[peak_locs]

        # Threshold peaks.
        th_peak_locs = peak_locs[peak_vals > threshold]
        th_peak_confidences = chunk_confidence[th_peak_locs]
        chunk_offset = chunk_duration * chunk_id
        th_peak_timestamps = chunk_offset + th_peak_locs/frame_rate
        n_peaks = len(th_peak_timestamps)
        logging.info("Number of timestamps: {}".format(n_peaks))

        # Export timestamps.
        event_times = event_times + list(th_peak_timestamps)
        event_confidences = event_confidences + list(th_peak_confidences)
        df = pd.DataFrame({
            "Time (s)": event_times,
            "Confidence (%)": event_confidences
        })
        df.to_csv(
            timestamps_path,
            columns=df_columns, float_format='%8.2f', index=True)

        if export_clips:
            for t in th_peak_timestamps:
                clip_start = max(0, int(np.round(sr*(t-0.5*clip_duration))))
                clip_stop = min(
                    len(sound_file), int(np.round(sr*(t+0.5*clip_duration))))
                sound_file.seek(clip_start)
                audio_clip = sound_file.read(clip_stop-clip_start)
                clip_name = suffix + "{:08.2f}".format(t).replace(".", "-")
                clip_path = get_output_path(
                    filepath, clip_name + ".wav", output_dir=clips_dir)
                librosa.output.write_wav(clip_path, audio_clip, sr)

    # Loop over chunks.
    for chunk_id in range(queue_length, n_chunks-1):
        # Print chunk ID and number of chunks.
        logging.info("Chunk ID: {}/{}".format(
            str(1+chunk_id).zfill(len(str(n_chunks))), n_chunks))

        # Read chunk.
        chunk_start = chunk_id * chunk_length
        sound_file.seek(chunk_start)
        chunk_audio = sound_file.read(chunk_length+chunk_padding)

        # Compute PCEN.
        deque.popleft()
        chunk_pcen = compute_pcen(chunk_audio, sr)
        deque.append(chunk_pcen)

        # Compute percentiles
        concat_deque = np.concatenate(deque, axis=1, out=concat_deque)
        deque_context = np.percentile(
            concat_deque, percentiles,
            axis=1, out=deque_context, overwrite_input=True)

        # Predict.
        if has_context:
            chunk_confidence = predict_with_context(
                chunk_pcen, deque_context, detector, logger_level,
                padding=chunk_padding)
        else:
            chunk_confidence = predict(
                chunk_pcen, detector, logger_level,
                padding=chunk_padding)

        # Remove trailing singleton dimension
        chunk_confidence = np.squeeze(chunk_confidence)

        # Map confidence to 0-100 range.
        chunk_confidence = map_confidence(chunk_confidence, detector_name)

        # If continuous confidence is required, store it in memory.
        if export_confidence:
            chunk_confidences.append(chunk_confidence)

        # If thresholding is not required, jump to next chunk.
        if threshold is None:
            continue

        # Find peaks.
        peak_locs, _ = scipy.signal.find_peaks(chunk_confidence)
        peak_vals = chunk_confidence[peak_locs]

        # Threshold peaks.
        th_peak_locs = peak_locs[peak_vals > threshold]
        th_peak_confidences = chunk_confidence[th_peak_locs]
        chunk_offset = chunk_duration * chunk_id
        th_peak_timestamps = chunk_offset + th_peak_locs/frame_rate
        n_peaks = len(th_peak_timestamps)
        logging.info("Number of timestamps: {}".format(n_peaks))

        # Export timestamps.
        event_times = event_times + list(th_peak_timestamps)
        event_confidences = event_confidences + list(th_peak_confidences)
        df = pd.DataFrame({
            "Time (s)": event_times,
            "Confidence (%)": event_confidences
        })
        df.to_csv(
            timestamps_path,
            columns=df_columns, float_format='%8.2f', index=True)

        # Export clips.
        if export_clips:
            for t in th_peak_timestamps:
                clip_start = max(0, int(np.round(sr*(t-0.5*clip_duration))))
                clip_stop = min(
                    len(sound_file), int(np.round(sr*(t+0.5*clip_duration))))
                sound_file.seek(clip_start)
                audio_clip = sound_file.read(clip_stop-clip_start)
                clip_name = suffix + "{:08.2f}".format(t).replace(".", "-")
                clip_path = get_output_path(
                    filepath, clip_name + ".wav", output_dir=clips_dir)
                librosa.output.write_wav(clip_path, audio_clip, sr)

    # Last chunk.
    # Print chunk ID and number of chunks.
    logging.info("Chunk ID: {}/{}".format(n_chunks, n_chunks))
    chunk_start = (n_chunks-1) * chunk_length
    sound_file.seek(chunk_start)
    chunk_audio = sound_file.read(full_length - chunk_start)
    chunk_pcen = compute_pcen(chunk_audio, sr)
    if has_context:
        # If the queue is empty, compute percentiles on the fly.
        if n_chunks == 1:
            deque_context = np.percentile(
                chunk_pcen, percentiles, axis=1, overwrite_input=True)

        # Predict.
        chunk_confidence = predict_with_context(
            chunk_pcen, deque_context, detector, logger_level,
            padding=0)
    else:
        # Predict.
        chunk_confidence = predict(
            chunk_pcen, detector, logger_level,
            padding=0)

    # Remove trailing singleton dimension
    chunk_confidence = np.squeeze(chunk_confidence)

    # Map confidence to 0-100 range.
    chunk_confidence = map_confidence(chunk_confidence, detector_name)

    # Threshold last chunk if required.
    if threshold is not None:

        # Find peaks.
        peak_locs, _ = scipy.signal.find_peaks(chunk_confidence)
        peak_vals = chunk_confidence[peak_locs]

        # Threshold peaks.
        th_peak_locs = peak_locs[peak_vals > threshold]
        th_peak_confidences = chunk_confidence[th_peak_locs]
        chunk_offset = chunk_duration * (n_chunks-1)
        th_peak_timestamps = chunk_offset + th_peak_locs/frame_rate
        n_peaks = len(th_peak_timestamps)
        logging.info("Number of timestamps: {}".format(n_peaks))

        # Export timestamps.
        event_times = event_times + list(th_peak_timestamps)
        event_confidences = event_confidences + list(th_peak_confidences)
        df = pd.DataFrame({
            "Time (s)": event_times,
            "Confidence (%)": event_confidences
        })
        df.to_csv(
            timestamps_path,
            columns=df_columns, float_format='%8.2f', index=True)

        # Export clips.
        if export_clips:
            for t in th_peak_timestamps:
                clip_start = max(0, int(np.round(sr*(t-0.5*clip_duration))))
                clip_stop = min(
                    len(sound_file), int(np.round(sr*(t+0.5*clip_duration))))
                sound_file.seek(clip_start)
                audio_clip = sound_file.read(clip_stop-clip_start)
                clip_name = suffix + "{:08.2f}".format(t).replace(".", "-")
                clip_path = get_output_path(
                    filepath, clip_name + ".wav", output_dir=clips_dir)
                librosa.output.write_wav(clip_path, audio_clip, sr)

    # Export confidence curve.
    if export_confidence:

        # Define output path for confidence.
        confidence_path = get_output_path(
            filepath, suffix + "confidence.hdf5", output_dir=output_dir)

        # Export confidence curve, chunk by chunk.
        # NB: looping over chunks, rather than concatenating them into a single
        # NumPy array, guarantees that this export has a O(1) memory footprint
        # with respect to the duration of the input file. As a result,
        # BirdVoxDetect is guaranteed to run with 3-4 gigabytes of RAM
        # for a context duration of 30 minutes, whatever be the duration
        # of the input.
        chunk_confidences.append(chunk_confidence)
        total_length = sum(map(len, chunk_confidences))
        with h5py.File(confidence_path, "w") as f:
            f.create_dataset('confidence', (total_length,), dtype="float32")
            f.create_dataset('time', (total_length,), dtype="float32")
            f["chunk_duration"] = chunk_duration
            f["frame_rate"] = frame_rate

        chunk_pointer = 0

        # Loop over chunks.
        for chunk_id, chunk_confidence in enumerate(chunk_confidences):

            # Define offset.
            chunk_length = len(chunk_confidence)
            next_chunk_pointer = chunk_pointer + chunk_length
            chunk_start = chunk_duration * chunk_id
            chunk_stop = chunk_start + chunk_length/frame_rate
            chunk_time = np.linspace(
                chunk_start, chunk_stop,
                num=chunk_length, endpoint=False, dtype=np.float32)

            # Export chunk as HDF5
            with h5py.File(confidence_path, "a") as f:
                f["confidence"][chunk_pointer:next_chunk_pointer] =\
                    chunk_confidence
                f["time"][chunk_pointer:next_chunk_pointer] = chunk_time

            # Increment pointer.
            chunk_pointer = next_chunk_pointer

    # Print final messages.
    logging.info("Done with file: {}.".format(filepath))
    if threshold is not None:
        timestamp_str = "Timestamps are available at: {}"
        logging.info(timestamp_str.format(timestamps_path))
    if export_clips:
        logging.info("Clips are available at: {}".format(clips_dir))
    if export_confidence:
        event_str = "Event detection curve is available at: {}"
        logging.info(event_str.format(confidence_path))


def compute_pcen(audio, sr):
    # Load settings.
    pcen_settings = get_pcen_settings()

    # Map to the range [-2**31, 2**31[
    audio = (audio * (2**31)).astype('float32')

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
    # NB: as of librosa v0.6.2, melspectrogram is type-instable and thus
    # returns 64-bit output even with a 32-bit input. Therefore, we need
    # to convert PCEN to single precision eventually. This might not be
    # necessary in the future, if the whole PCEN pipeline is kept type-stable.
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
        bias=pcen_settings["pcen_delta"],
        power=pcen_settings["pcen_power"],
        time_constant=pcen_settings["pcen_time_constant"])

    # Convert to single floating-point precision.
    pcen = pcen.astype('float32')

    # Truncate spectrum to range 2-10 kHz.
    pcen = pcen[:pcen_settings["top_freq_id"], :]

    # Return.
    return pcen


def predict(pcen, detector, logger_level, padding=0):
    pcen_settings = get_pcen_settings()

    # PCEN-SNR
    if detector == "pcen_snr":
        frame_rate = 20.0
        pcen_snr = np.max(pcen, axis=0) - np.min(pcen, axis=0)
        pcen_confidence = pcen_snr / (0.001 + pcen_snr)
        median_confidence = scipy.signal.medfilt(
            pcen_confidence, kernel_size=127)
        sr = pcen_settings["sr"]
        hop_length = pcen_settings["hop_length"]
        audio_duration = pcen.shape[1]*hop_length/sr
        confidence_x = np.arange(
            0.0,
            audio_duration*sr/hop_length,
            sr/(hop_length*frame_rate))[:-1].astype('int')
        y = 100 * np.clip(median_confidence[confidence_x], 0.0, 1.0)

    # PCEN-CNN. (no context adaptation)
    else:
        # Compute number of hops.
        clip_length = 104
        pcen_settings = get_pcen_settings()
        stride_length = pcen_settings["stride_length"]
        n_freqs, n_padded_hops = pcen.shape
        if padding > 0:
            padding_hops = int(padding / pcen_settings["hop_length"])
            n_hops = n_padded_hops - padding_hops + 1
            n_strides = int(n_hops / stride_length)
        else:
            n_hops = n_padded_hops
            n_strides = int((n_hops - clip_length) / stride_length)
        itemsize = pcen.itemsize

        # Stride and tile.
        X_shape = (n_strides, clip_length, n_freqs)
        X_stride = (itemsize*n_freqs*stride_length, itemsize*n_freqs, itemsize)
        X_pcen = np.lib.stride_tricks.as_strided(
            np.ravel(np.copy(pcen).T),
            shape=X_shape,
            strides=X_stride,
            writeable=False)
        X_pcen = np.transpose(X_pcen, (0, 2, 1))[:, :, :, np.newaxis]

        # Predict.
        verbose = (logger_level < 15)
        y = detector.predict(X_pcen, verbose=verbose)

    return y


def predict_with_context(pcen, context, detector, logger_level, padding=0):
    # Truncate frequency spectrum (from 128 to 120 bins)
    pcen = pcen[:120, :]
    context = context[:, :120]

    # Compute number of hops.
    clip_length = 104
    pcen_settings = get_pcen_settings()
    stride_length = pcen_settings["stride_length"]
    n_freqs, n_padded_hops = pcen.shape
    if padding > 0:
        padding_hops = int(padding / pcen_settings["hop_length"])
        n_hops = n_padded_hops - padding_hops + 1
        n_strides = int(n_hops / stride_length)
    else:
        n_hops = n_padded_hops
        n_strides = int((n_hops - clip_length) / stride_length)
    itemsize = pcen.itemsize

    # Stride and tile.
    X_shape = (n_strides, clip_length, n_freqs)
    X_stride = (itemsize*n_freqs*stride_length, itemsize*n_freqs, itemsize)
    X_pcen = np.lib.stride_tricks.as_strided(
        np.ravel(np.copy(pcen).T),
        shape=X_shape,
        strides=X_stride,
        writeable=False)
    X_pcen = np.transpose(X_pcen, (0, 2, 1))[:, :, :, np.newaxis]
    X_bg = np.broadcast_to(
        context.T[np.newaxis, :, :],
        (n_strides, context.shape[1], context.shape[0]))

    # Predict.
    verbose = (logger_level < 15)
    y = detector.predict(
        {"spec_input": X_pcen, "bg_input": X_bg},
        verbose=verbose)

    # Return confidence.
    return y


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
        "stride_length": 34,
        "top_freq_id": 128,
        "win_length": 256,
        "window": "flattop"}
    return pcen_settings


def get_model_path(model_name):
    return os.path.join(
        os.path.dirname(__file__), "models", model_name)


def map_confidence(y, model_name):
    if model_name == "birdvoxdetect_pcen_cnn_adaptive-threshold-T1800":
        # Calibrated on BirdVox-300h.
        # See repository: github.com/BirdVox/birdvox-full-season
        # Notebook: detector/notebooks/BirdVoxDetect-v01_calibration.ipynb
        # This model encodes "0" resp. "1" as "event" resp. "no event"
        log1my = np.log1p(np.clip(-y, np.finfo(np.float32).eps - 1, None))
        logy = np.log(np.clip(y, np.finfo(np.float32).tiny, None))
        y_inverse_sigmoid = log1my - logy
        y_out = 2.7 * y_inverse_sigmoid - 21
    elif model_name == "birdvoxdetect_pcen_cnn":
        # Calibrated on BirdVox-full-night_unit03_00-19-45_01min.
        # See repository: github.com/BirdVox/birdvox-full-season
        # Notebook: detector/notebooks/BirdVoxDetect-v01_calibration-nocontext.ipynb
        # This model encodes "1" resp. "0" as "event" resp. "no event"
        log1my = np.log1p(np.clip(-y, np.finfo(np.float32).eps - 1, None))
        logy = np.log(np.clip(y, np.finfo(np.float32).tiny, None))
        y_inverse_sigmoid = logy - log1my
        y_out = 6 * y_inverse_sigmoid
    else:
        y_out = y
    return y_out
