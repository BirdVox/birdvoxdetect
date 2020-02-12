import birdvoxclassify
import collections
from contextlib import redirect_stderr
import datetime
import h5py
import joblib
import json
import librosa
import logging
import numpy as np
import operator
import os
import pandas as pd
import scipy
import scipy.signal
import sklearn
import soundfile as sf
import traceback
import warnings

# Width of the spectrogram matrix as input to the convnets
BVD_CLIP_LENGTH = 104


with warnings.catch_warnings():
    # Suppress TF and Keras warnings when importing
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.simplefilter("ignore")
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    with redirect_stderr(open(os.devnull, "w")):
        from tensorflow import keras


import birdvoxdetect
from birdvoxdetect.birdvoxdetect_exceptions import BirdVoxDetectError


def map_tfr(x_tfr):
    return (np.log1p(0.5 * x_tfr) - 0.8)


def process_file(
        filepath,
        output_dir=None,
        export_clips=False,
        export_confidence=False,
        threshold=50.0,
        suffix="",
        clip_duration=1.0,
        logger_level=logging.INFO,
        detector_name="birdvoxdetect-v03_trial-12_network_epoch-068",
        classifier_name="_".join([
            "birdvoxclassify-flat-multitask-convnet",
            "tv1hierarchical-2e7e1bbd434a35b3961e315cfe3832fc"]),
        custom_objects=None,
        bva_threshold=0.5):

    # Set logger level.
    logger = logging.getLogger("logger_stream")
    logger.setLevel(logger_level)

    # Print new line and file name.
    logger.info("-" * 80)
    modules = [
        birdvoxdetect, birdvoxclassify, h5py, joblib, json,
        librosa, logging, np, pd, tf, scipy, sf, sklearn]
    for module in modules:
        logger.debug(module.__name__.ljust(15) + " v" + module.__version__)
    logger.info("")
    logger.info("Loading file: {}".format(filepath))

    # Check for existence of the input file.
    if not os.path.exists(filepath):
        raise BirdVoxDetectError(
            'File "{}" could not be found.'.format(filepath))

    # Load the file.
    try:
        sound_file = sf.SoundFile(filepath)
    except Exception:
        logger.warn('Could not open file: {}'.format(filepath))
        return

    # Load the detector of sensor faults.
    sensorfault_detector_name = 'birdvoxactivate.pkl'
    logger.info("Sensor fault detector: {}".format(sensorfault_detector_name))
    sensorfault_model_path = get_model_path(sensorfault_detector_name)
    if not os.path.exists(sensorfault_model_path):
        raise BirdVoxDetectError(
            'Model "{}" could not be found.'.format(sensorfault_model_name))
    sensorfault_model = joblib.load(sensorfault_model_path)

    # Load the detector of flight calls.
    logger.info("Flight call detector: {}".format(detector_name))
    if detector_name == "pcen_snr":
        detector = "pcen_snr"
    else:
        detector_model_path = get_model_path(detector_name + '.h5')
        if not os.path.exists(detector_model_path):
            raise BirdVoxDetectError(
                'Model "{}" could not be found.'.format(detector_name))
        MAX_LOAD_ATTEMPTS = 10
        load_attempt_id = 0
        is_load_successful = False
        while not is_load_successful and (load_attempt_id<MAX_LOAD_ATTEMPTS):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    detector = keras.models.load_model(
                        detector_model_path, custom_objects=custom_objects)
                is_load_successful = True
            except Exception:
                load_attempt_id += 1
        if not is_load_successful:
            exc_str = 'Could not open detector model "{}":\n{}'
            formatted_trace = traceback.format_exc()
            exc_formatted_str = exc_str.format(
                detector_model_path, formatted_trace)
            raise BirdVoxDetectError(exc_formatted_str)

    # Load the species classifier.
    logger.info("Species classifier: {}".format(classifier_name))
    classifier_model_path = birdvoxclassify.get_model_path(classifier_name)
    if not os.path.exists(classifier_model_path):
        raise BirdVoxDetectError(
            'Model "{}" could not be found.'.format(classifier_name))
    load_attempt_id = 0
    is_load_successful = False
    while not is_load_successful and (load_attempt_id<MAX_LOAD_ATTEMPTS):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                classifier = keras.models.load_model(
                    classifier_model_path, custom_objects=custom_objects)
            is_load_successful = True
        except Exception:
            load_attempt_id += 1
    if not is_load_successful:
        exc_str = 'Could not open classifier model "{}":\n{}'
        formatted_trace = traceback.format_exc()
        exc_formatted_str = exc_str.format(
            classifier_model_path, formatted_trace)
        raise BirdVoxDetectError(exc_formatted_str)

    # Load the taxonomy.
    taxonomy_path = birdvoxclassify.get_taxonomy_path(classifier_name)
    with open(taxonomy_path) as f:
        taxonomy = json.load(f)

    # Define percentiles.
    percentiles = [0.1, 1, 10, 25, 50, 75, 90, 99, 99.9]

    # Define chunk size.
    has_context = "T-" in detector_name
    if has_context:
        queue_length = 4
        T_str = detector_name.split("T-")[1].split("_")[0]
        chunk_duration = int(T_str) / queue_length
    else:
        chunk_duration = 450
        queue_length = 1

    # Define minimum peak height for BirdVoxDetect function.
    min_peak_height = min(10, threshold)

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

    # Initialize checklist as a Pandas DataFrame.
    if threshold is not None:
        checklist_path = get_output_path(
            filepath, suffix + "checklist.csv", output_dir=output_dir)
        event_hhmmss = []
        event_4lettercodes = []
        event_confidences = []
        df_columns = [
            "Time (hh:mm:ss)", "Species (4-letter code)", "Confidence (%)"]
        df = pd.DataFrame({
            "Time (hh:mm:ss)": event_hhmmss,
            "Species (4-letter code)": event_4lettercodes,
            "Confidence (%)": event_confidences
        })
        df.to_csv(checklist_path, columns=df_columns, index=False)

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
    logger.debug("Chunk duration: {} seconds".format(chunk_duration))

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
    if (n_chunks>1):
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
            sensorfault_features)[0]

        # If probability of sensor fault is above 50%,
        # exclude start of recording
        if sensor_fault_probability > bva_threshold:
            logger.debug("Probability of sensor fault: {:5.2f}%".format(
                100*sensor_fault_probability))
            chunk_id_start = min(n_chunks-1, queue_length)
            context_duration = chunk_duration
            context_duration_str = str(datetime.timedelta(
                seconds=context_duration))
            logger.debug(
                "Ignoring segment between 00:00:00 and " +\
                context_duration_str + " (" + str(chunk_id_start) +\
                " chunks)")
            # If continuous confidence is required, store it in memory.
            if export_confidence:
                chunk_confidence = np.full(
                    int(chunk_id_start * chunk_duration * frame_rate), np.nan)
                chunk_confidences.append(chunk_confidence)
        else:
            chunk_id_start = 0
    else:
        chunk_id_start = 0
        has_sensor_fault = False

    # Define frame rate.
    frame_rate = pcen_settings["sr"] /\
        (pcen_settings["hop_length"] * pcen_settings["stride_length"])

    # Compute confidence on queue chunks.
    # NB: the following loop is skipped if there is a single chunk.
    for chunk_id in range(chunk_id_start, min(queue_length, n_chunks-1)):
        # Print chunk ID and number of chunks.
        logger.debug("Chunk ID: {}/{}".format(
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

        # Map confidence to 0-100 range.
        chunk_confidence = map_confidence(chunk_confidence, detector_name)

        # If continuous confidence is required, store it in memory.
        if export_confidence:
            chunk_confidences.append(chunk_confidence)

        # If thresholding is not required, jump to next chunk.
        if threshold is None:
            continue

        # Find peaks.
        peak_locs, _ = scipy.signal.find_peaks(
            chunk_confidence, height=min_peak_height)
        peak_vals = chunk_confidence[peak_locs]

        # Threshold peaks.
        th_peak_locs = peak_locs[peak_vals > threshold]
        th_peak_confidences = chunk_confidence[th_peak_locs]
        chunk_offset = chunk_duration * chunk_id
        chunk_timestamps = chunk_offset + th_peak_locs/frame_rate
        n_peaks = len(chunk_timestamps)

        # Classify species.
        th_peak_4lettercodes = list(map(
            lambda x: classify_species(classifier, chunk_pcen, x, taxonomy),
            th_peak_locs))

        # Count flight calls.
        chunk_counter = collections.Counter(th_peak_4lettercodes)
        logger.debug(
            "Number of flight calls in current chunk: {}".format(n_peaks))
        logger.debug("(" + ", ".join((str(v) + " " + k)
            for (k, v) in chunk_counter.most_common()) + ")")

        # Export timestamps.
        chunk_hhmmss = list(map(seconds_to_hhmmss, chunk_timestamps))
        event_hhmmss = event_hhmmss + chunk_hhmmss
        chunk_4lettercodes = list(th_peak_4lettercodes)
        event_4lettercodes = event_4lettercodes + chunk_4lettercodes
        event_confidences = event_confidences + list(th_peak_confidences)
        df = pd.DataFrame({
            "Time (hh:mm:ss)": event_hhmmss,
            "Species (4-letter code)": event_4lettercodes,
            "Confidence (%)": event_confidences
        })
        df.to_csv(checklist_path, columns=df_columns, index=False)

        # Export clips.
        if export_clips:
            chunk_zip = zip(
                chunk_timestamps, chunk_hhmmss,
                list(th_peak_confidences), chunk_4lettercodes)
            for clip_timestamp, clip_hhmmss, clip_confidence, clip_4lettercode in chunk_zip:
                clip_start = max(0, int(np.round(
                    sr*(clip_timestamp-0.5*clip_duration))))
                clip_stop = min(
                    len(sound_file), int(np.round(
                    sr*(clip_timestamp+0.5*clip_duration))))
                sound_file.seek(clip_start)
                audio_clip = sound_file.read(clip_stop-clip_start)
                clip_hhmmss_escaped = clip_hhmmss.replace(
                    ":", "_").replace(".", "-")
                clip_name = suffix + "_".join([
                    clip_hhmmss_escaped,
                    str(int(clip_confidence)), clip_4lettercode])
                clip_path = get_output_path(
                    filepath, clip_name + ".wav", output_dir=clips_dir)
                sf.write(clip_path, audio_clip, sr)

    # Loop over chunks.
    chunk_id = queue_length
    while chunk_id < (n_chunks-1):
        # Print chunk ID and number of chunks.
        logger.debug("Chunk ID: {}/{}".format(
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

        # Compute sensor fault features.
        # Median is 4th order statistic. Restrict to lowest 120 mel-freq bins
        context_median = deque_context[4, :120]
        context_median_medfilt = scipy.signal.medfilt(
        context_median, kernel_size=(13,))
        sensorfault_features = context_median_medfilt[::12].reshape(1, -1)

        # Compute probability of sensor fault.
        sensor_fault_probability =\
            sensorfault_model.predict(sensorfault_features)[0]

        # If probability of sensor fault is above threshold,
        # exclude start of recording
        has_sensor_fault = (sensor_fault_probability > bva_threshold)
        if has_sensor_fault:
            logger.debug("Probability of sensor fault: {:5.2f}%".format(
                100*sensor_fault_probability))
            context_duration = queue_length * chunk_duration
            ignored_start_str = str(datetime.timedelta(
                seconds=chunk_id*chunk_duration))
            ignored_stop_str = str(datetime.timedelta(
                seconds=(chunk_id+1)*chunk_duration))
            logger.debug(
                "Ignoring segment between " +\
                ignored_start_str + " and " +\
                ignored_stop_str + " (1 chunk)")
            if export_confidence:
                chunk_confidence_length =\
                    int(queue_length * chunk_duration * frame_rate)
                chunk_confidences.append(np.full(
                    chunk_confidence_length, np.nan))
            chunk_id = chunk_id + 1
            continue

        # Otherwise, detect flight calls.
        if has_context:
            chunk_confidence = predict_with_context(
                chunk_pcen, deque_context, detector, logger_level,
                padding=chunk_padding)
        else:
            chunk_confidence = predict(
                chunk_pcen, detector, logger_level,
                padding=chunk_padding)

        # Map confidence to 0-100 range.
        chunk_confidence = map_confidence(chunk_confidence, detector_name)

        # If continuous confidence is required, store it in memory.
        if export_confidence:
            chunk_confidences.append(chunk_confidence)

        # If thresholding is not required, jump to next chunk.
        if threshold is None:
            continue

        # Find peaks.
        peak_locs, _ = scipy.signal.find_peaks(
            chunk_confidence, height=min_peak_height)
        peak_vals = chunk_confidence[peak_locs]

        # Threshold peaks.
        th_peak_locs = peak_locs[peak_vals > threshold]
        th_peak_confidences = chunk_confidence[th_peak_locs]
        chunk_offset = chunk_duration * chunk_id
        chunk_timestamps = chunk_offset + th_peak_locs/frame_rate
        n_peaks = len(chunk_timestamps)

        # Classify species.
        th_peak_4lettercodes = list(map(
            lambda x: classify_species(classifier, chunk_pcen, x, taxonomy),
            th_peak_locs))
        chunk_4lettercodes = list(th_peak_4lettercodes)
        event_4lettercodes = event_4lettercodes + chunk_4lettercodes

        # Count flight calls.
        chunk_counter = collections.Counter(th_peak_4lettercodes)
        logger.debug(
            "Number of flight calls in current chunk: {}".format(n_peaks))
        logger.debug("(" + ", ".join((str(v) + " " + k)
            for (k, v) in chunk_counter.most_common()) + ")")

        # Export timestamps.
        chunk_hhmmss = list(map(seconds_to_hhmmss, chunk_timestamps))
        event_hhmmss = event_hhmmss + chunk_hhmmss
        event_confidences = event_confidences + list(th_peak_confidences)
        df = pd.DataFrame({
            "Time (hh:mm:ss)": event_hhmmss,
            "Species (4-letter code)": event_4lettercodes,
            "Confidence (%)": event_confidences
        })
        df.to_csv(checklist_path, columns=df_columns, index=False)

        # Export clips.
        if export_clips:
            chunk_zip = zip(
                chunk_timestamps, chunk_hhmmss,
                list(th_peak_confidences), chunk_4lettercodes)
            for clip_timestamp, clip_hhmmss, clip_confidence, clip_4lettercode in chunk_zip:
                clip_start = max(0, int(np.round(
                    sr*(clip_timestamp-0.5*clip_duration))))
                clip_stop = min(
                    len(sound_file), int(np.round(
                    sr*(clip_timestamp+0.5*clip_duration))))
                sound_file.seek(clip_start)
                audio_clip = sound_file.read(clip_stop-clip_start)
                clip_hhmmss_escaped = clip_hhmmss.replace(
                    ":", "_").replace(".", "-")
                clip_name = suffix + "_".join([
                    clip_hhmmss_escaped,
                    str(int(clip_confidence)), clip_4lettercode])
                clip_path = get_output_path(
                    filepath, clip_name + ".wav", output_dir=clips_dir)
                sf.write(clip_path, audio_clip, sr)

        # Go to next chunk.
        chunk_id = chunk_id + 1

    # Last chunk. For n_chunks>1, we reuse the context from the penultimate
    # chunk because this last chunk is typically shorter than chunk_length.
    # But if the queue is empty (n_chunks==1), we compute context on the fly
    # even if this chunk is shorter. This can potentially be numerically
    # unstable with files shorter than 30 minutes, which is why we issue a
    # warning. Also, we do not try to detect sensor faults in files shorter than
    # 30 minutes.
    if (n_chunks>1) and has_sensor_fault:
        logger.debug("Probability of sensor fault: {:5.2f}%".format(
            100*sensor_fault_probability))
        ignored_start_str = str(datetime.timedelta(
            seconds=chunk_id*chunk_duration))
        ignored_stop_str = str(datetime.timedelta(
            seconds=full_length*sr))
        logger.debug(
            "Ignoring segment between " +\
            segment_start_str + " and " +\
            segment_stop_str + " (i.e., up to end of file)")
    else:
        logger.debug("Chunk ID: {}/{}".format(n_chunks, n_chunks))
        chunk_start = (n_chunks-1) * chunk_length
        sound_file.seek(chunk_start)
        context_duration = queue_length * chunk_duration
        chunk_audio = sound_file.read(full_length - chunk_start)
        chunk_pcen = compute_pcen(chunk_audio, sr)
        chunk_confidence_length = int(frame_rate*full_length/sr)
        chunk_confidence = np.full(chunk_confidence_length, np.nan)

        if has_context and (n_chunks==1):
            deque_context = np.percentile(
                chunk_pcen, percentiles, axis=1, overwrite_input=True)
            logging.warning(
                "\nFile duration (" + str(datetime.timedelta(seconds=full_length/sr)) +\
                ") shorter than 25% of context duration (" +\
                str(datetime.timedelta(seconds=context_duration)) + ").\n"
                "This may cause numerical instabilities in threshold adaptation.\n" +\
                "We recommend disabling the context-adaptive threshold\n" +\
                "(i.e., setting 'detector_name'='birdvoxdetect-v03_trial-12_network_epoch-06') when\n" +\
                "running birdvoxdetect on short audio files.")
            has_sensor_fault = False
        elif has_context:
            # Compute percentiles
            deque.popleft()
            deque.append(chunk_pcen)
            concat_deque = np.concatenate(deque, axis=1, out=concat_deque)
            deque_context = np.percentile(
                concat_deque, percentiles,
                axis=1, out=deque_context, overwrite_input=True)


    if not has_sensor_fault:
        # Define trimming length for last chunk.
        # This one is not equal to one second but to the duration
        # of a BVD/BVC clip, i.e. about 150 milliseconds.
        # Note that this trimming is not compensated by the presence of
        # the next chunk because we are already at the last chunk.
        # In other words, if a flight call happens at the last 150 milliseconds
        # of an audio recording, it is ignored.
        lastchunk_trimming = BVD_CLIP_LENGTH * pcen_settings["hop_length"]
        if has_context:
            # Predict.
            chunk_confidence = predict_with_context(
                chunk_pcen, deque_context, detector, logger_level,
                padding=lastchunk_trimming)
        else:
            # Predict.
            chunk_confidence = predict(
                chunk_pcen, detector, logger_level, padding=lastchunk_trimming)

        # Map confidence to 0-100 range.
        chunk_confidence = map_confidence(chunk_confidence, detector_name)

        # Threshold last chunk if required.
        if threshold is not None:

            # Find peaks.
            peak_locs, _ = scipy.signal.find_peaks(
                chunk_confidence, height=min_peak_height)
            peak_vals = chunk_confidence[peak_locs]

            # Threshold peaks.
            th_peak_locs = peak_locs[peak_vals > threshold]
            th_peak_confidences = chunk_confidence[th_peak_locs]

            chunk_offset = chunk_duration * (n_chunks-1)
            chunk_timestamps = chunk_offset + th_peak_locs/frame_rate
            n_peaks = len(chunk_timestamps)

            # Classify species.
            th_peak_4lettercodes = list(map(
                lambda x: classify_species(classifier, chunk_pcen, x, taxonomy),
                th_peak_locs))

            # Count flight calls.
            chunk_counter = collections.Counter(th_peak_4lettercodes)
            logger.debug(
                "Number of flight calls in current chunk: {}".format(n_peaks))
            logger.debug("(" + ", ".join((str(v) + " " + k)
                for (k, v) in chunk_counter.most_common()) + ")")

            # Export timestamps.
            chunk_hhmmss = list(map(seconds_to_hhmmss, chunk_timestamps))
            event_hhmmss = event_hhmmss + chunk_hhmmss
            chunk_4lettercodes = list(th_peak_4lettercodes)
            event_4lettercodes = event_4lettercodes + chunk_4lettercodes
            event_confidences = event_confidences + list(th_peak_confidences)
            df = pd.DataFrame({
                "Time (hh:mm:ss)": event_hhmmss,
                "Species (4-letter code)": event_4lettercodes,
                "Confidence (%)": event_confidences
            })
            df.to_csv(checklist_path, columns=df_columns, index=False)

            # Export clips.
            if export_clips:
                chunk_zip = zip(
                    chunk_timestamps, chunk_hhmmss,
                    list(th_peak_confidences), chunk_4lettercodes)
                for clip_timestamp, clip_hhmmss, clip_confidence, clip_4lettercode in chunk_zip:
                    clip_start = max(0, int(np.round(
                        sr*(clip_timestamp-0.5*clip_duration))))
                    clip_stop = min(
                        len(sound_file), int(np.round(
                        sr*(clip_timestamp+0.5*clip_duration))))
                    sound_file.seek(clip_start)
                    audio_clip = sound_file.read(clip_stop-clip_start)
                    clip_hhmmss_escaped = clip_hhmmss.replace(
                        ":", "_").replace(".", "-")
                    clip_name = suffix + "_".join([
                        clip_hhmmss_escaped,
                        str(int(clip_confidence)), clip_4lettercode])
                    clip_path = get_output_path(
                        filepath, clip_name + ".wav", output_dir=clips_dir)
                    sf.write(clip_path, audio_clip, sr)

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
    if threshold is not None:
        df = pd.read_csv(checklist_path)
        logger.info("\n".join([(k + " " + str(v).rjust(6)) for (k, v) in
            collections.Counter(df["Species (4-letter code)"]).most_common()]))
        logger.info("TOTAL: {}.".format(str(len(df)).rjust(4)))
        timestamp_str = "Checklist is available at: {}"
        logger.info(timestamp_str.format(checklist_path))
    if export_clips:
        logger.info("Clips are available at: {}".format(clips_dir))
    if export_confidence:
        event_str = "Event detection curve is available at: {}"
        logger.info(event_str.format(confidence_path))
    logger.info("Done with file: {}.".format(filepath))

    return df


def classify_species(classifier, chunk_pcen, th_peak_loc, taxonomy):
    # Load settings
    pcen_settings = get_pcen_settings()
    clip_length = BVD_CLIP_LENGTH

    # Convert birdvoxdetect hops to PCEN hops
    th_peak_hop = th_peak_loc * pcen_settings["stride_length"]

    # Extract clip in PCEN domain
    pcen_clip_start = th_peak_hop - clip_length//2
    pcen_clip_stop = th_peak_hop + clip_length//2
    pcen_clip = chunk_pcen[:120,
        pcen_clip_start:pcen_clip_stop, np.newaxis]

    # Call birdvoxclassify to extract rich prediction
    full_pred = birdvoxclassify.format_pred(
        birdvoxclassify.predict(pcen_clip, classifier=classifier),
        taxonomy=taxonomy)

    # Extract three-digit tag of maximum probability
    fine_tag = max({k: full_pred["fine"][k]["probability"]
            for k in full_pred["fine"]}.items(),
        key=operator.itemgetter(1))[0]

    # Convert three-digit tag to fine taxonomy
    aliases = full_pred["fine"][fine_tag]["taxonomy_level_aliases"]
    if "species_4letter_code" in aliases:
        event_4lettercode = aliases["species_4letter_code"]
    else:
        event_4lettercode = "OTHE"

    return event_4lettercode


def compute_pcen(audio, sr):
    # Load settings.
    pcen_settings = get_pcen_settings()

    # Validate audio
    librosa.util.valid_audio(audio, mono=True)

    # Map to the range [-2**31, 2**31[
    audio = (audio * (2**31)).astype('float32')

    # Resample to 22,050 Hz
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
    # Truncate frequency spectrum (from 128 to 120 bins)
    pcen = pcen[:120, :]
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
        clip_length = BVD_CLIP_LENGTH
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
            np.ravel(map_tfr(pcen).T),
            shape=X_shape,
            strides=X_stride,
            writeable=False)
        X_pcen = np.transpose(X_pcen, (0, 2, 1))[:, :, :, np.newaxis]

        # Predict.
        verbose = (logger_level < 15)
        y = detector.predict(X_pcen, verbose=verbose)

    return np.squeeze(y)


def predict_with_context(pcen, context, detector, logger_level, padding=0):
    # Truncate frequency spectrum (from 128 to 120 bins)
    pcen = pcen[:120, :]
    context = context[:, :120]

    # Compute number of hops.
    clip_length = BVD_CLIP_LENGTH
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
        np.ravel(map_tfr(pcen).T),
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
    return y.squeeze()


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
    log1my = np.log1p(np.clip(-y, np.finfo(np.float32).eps - 1, None))
    logy = np.log(np.clip(y, np.finfo(np.float32).tiny, None))
    y_inverse_sigmoid = log1my - logy - np.log(np.finfo(np.float32).eps)
    if model_name == "birdvoxdetect-v03_trial-12_network_epoch-068":
        # See birdvox-full-season/detector-v03/notebooks/07_measure-precision-300h.ipynb
        y_in = np.maximum(0, y_inverse_sigmoid-18)**2 / 100
        y_out = 14.76561354 * (y_in**3) - 68.54604756 * (y_in**2) +\
            111.89379155 * (y_in) - 0.13061346
    elif model_name == "birdvoxdetect-v03_T-1800_trial-37_network_epoch-023":
        # See birdvox-full-season/detector-v03/notebooks/10_measure-precision-300h-ca.ipynb
        y_in = np.maximum(0, y_inverse_sigmoid-18)**2 / 100
        y_out = 4.28734484 * (y_in**3) - 25.97219728 * (y_in**2) +\
            62.66749547 * (y_in) + 4.8942351
    else:
        y_in = y_inverse_sigmoid**2
        y_out = 0.09*y_in
    return np.clip(y_out, 0.0, 99.99)


def seconds_to_hhmmss(total_seconds):
    hours, tmp_seconds = divmod(total_seconds, 3600)
    minutes, seconds = divmod(tmp_seconds, 60)
    return "{:02d}:{:02d}:{:05.2f}".format(int(hours), int(minutes), seconds)
