# combines videos with matching audio files (e.g. audio descriptions)
# input: video or folder of videos and an audio file or folder of audio files
# output: videos in a folder "videos_with_ad", with aligned segments of the audio replaced
# this script aligns the new audio to the video using the video's old audio

'''
Copyright (C) 2023  Julian Brown

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

VIDEO_EXTENSIONS = set(['mp4', 'mkv', 'avi', 'mov', 'webm', 'm4v', 'flv', 'vob'])
AUDIO_EXTENSIONS = set(['mp3', 'm4a', 'opus', 'wav', 'aac', 'flac', 'ac3', 'mka'])
PLOT_ALIGNMENT_TO_FILE = True

TIMESTEPS_PER_SECOND = 10  # factors must be subset of (2, 3, 5, 7)
TIMESTEP_SIZE_SECONDS = 1. / TIMESTEPS_PER_SECOND
AUDIO_SAMPLE_RATE = 44100
DITHER_PERIOD_STEPS = 10
MAX_RATE_RATIO_DIFF_ALIGN = .1
MIN_DURATION_TO_REPLACE_SECONDS = 2
JUST_NOTICEABLE_DIFF_IN_FREQ_RATIO = .005
MIN_STRETCH_OFFSET = 30

if PLOT_ALIGNMENT_TO_FILE:
  import matplotlib.pyplot as plt
import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import os
import glob
import itertools
from pathlib import Path
import sys
from typing import Optional
import numpy as np
import ffmpeg
import platformdirs
import static_ffmpeg
import scipy.signal
import scipy.optimize
import scipy.interpolate
import scipy.sparse
import configparser
import traceback
import multiprocessing
import platform
import natsort
from collections import defaultdict
from sortedcontainers import SortedList
import hashlib

try:
  import wx
  gui_font = (11, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, "Arial")
except ImportError:
  wx = None

gui_update_interval_ms = 100
gui_background_color_dark = (28, 30, 35)
gui_background_color_light = (170, 182, 211)

IS_RUNNING_WINDOWS = platform.system() == 'Windows'
if IS_RUNNING_WINDOWS:
  default_output_dir = 'videos_with_ad'
  default_alignment_dir = 'alignment_plots'
else:
  default_output_dir = os.path.expanduser('~') + '/videos_with_ad'
  default_alignment_dir = os.path.expanduser('~') + '/alignment_plots'

def ensure_folders_exist(dirs):
  for dir in dirs:
    if not os.path.isdir(dir):
      print(f"Directory not found, creating it: {dir}")
      os.makedirs(dir)

def get_sorted_filenames(path, extensions, alt_extensions=set([])):
  # path could be three different things: a file, a directory, a list of files
  if type(path) is list:
    files = [os.path.abspath(file) for file in path]
    for file in files:
      if not os.path.isfile(file):
        raise RuntimeError(f"No file found at input path:\n  {file}")
  else:
    path = os.path.abspath(path)
    if os.path.isdir(path):
      files = glob.glob(glob.escape(path) + "/*")
      if len(files) == 0:
        raise RuntimeError(f"Empty input directory:\n  {path}")
    else:
      if not os.path.isfile(path):
        raise RuntimeError(f"No file or directory found at input path:\n  {path}")
      files = [path]
  files = [file for file in files if os.path.splitext(file)[1][1:] in extensions | alt_extensions]
  if len(files) == 0:
    error_msg = [f"No files with valid extensions found at input path:\n  {path}",
                 "Did you accidentally put the audio filepath before the video filepath?",
                 "The video path should be the first positional input, audio second.",
                 "Or maybe you need to add a new extension to this script's regex?",
                 f"valid extensions for this input are:\n  {extensions}"]
    raise RuntimeError("\n".join(error_msg))
  files = natsort.os_sorted(files)
  has_alt_extensions = [0 if os.path.splitext(file)[1][1:] in extensions else 1 for file in files]
  return files, has_alt_extensions

# ffmpeg command error handler
def run_ffmpeg_command(command, err_msg):
  try:
    return command.run(capture_stdout=True, capture_stderr=True, cmd=get_ffmpeg())
  except ffmpeg.Error as e:
    print("  ERROR: ffmpeg failed to " + err_msg)
    print("FFmpeg error:")
    print(e.stderr.decode('utf-8'))
    raise

def run_async_ffmpeg_command(command, media_arr, err_msg):
  try:
    ffmpeg_caller = command.run_async(pipe_stdin=True, quiet=True, cmd=get_ffmpeg())
    out, err = ffmpeg_caller.communicate(media_arr.astype(np.int16).T.tobytes())
    if len(err) > 0:
      print("  ERROR: ffmpeg failed to " + err_msg)
      print("FFmpeg error:")
      print(err.decode('utf-8'))
      raise ChildProcessError('FFmpeg error.')
  except ffmpeg.Error as e:
    print("  ERROR: ffmpeg failed to " + err_msg)
    print("FFmpeg error:")
    print(e.stderr.decode('utf-8'))
    raise

# read audio from file with ffmpeg and convert to numpy array
def parse_audio_from_file(media_file, num_channels=2):
  # retrieve only the first audio track, injecting silence/trimming to force timestamps to match up
  # for example, when the video starts before the audio this fills that starting gap with silence
  ffmpeg_command = ffmpeg.input(media_file).output('-', format='s16le', acodec='pcm_s16le',
                                                   af='aresample=async=1:first_pts=0', map='0:a:0',
                                                   ac=num_channels, ar=AUDIO_SAMPLE_RATE, loglevel='error')
  media_stream, _ = run_ffmpeg_command(ffmpeg_command, f"parse audio from input file: {media_file}")
  # media_arr = np.frombuffer(media_stream, np.int16).astype(np.float32).reshape((-1, num_channels)).T
  media_arr = np.frombuffer(media_stream, np.int16).astype(np.float16).reshape((-1, num_channels)).T
  return media_arr

def plot_alignment(plot_filename_no_ext, path, audio_times, video_times, similarity_percent,
                   median_slope, stretch_audio, no_pitch_correction):
  downsample = 20
  path = path[::downsample]
  video_times_full, audio_times_full, cluster_indices, quals, cum_quals = path.T
  scatter_color = [.2,.4,.8]
  lcs_rgba = np.zeros((len(quals),4))
  lcs_rgba[:,:3] = np.array(scatter_color)[None,:]
  lcs_rgba[:,3] = np.clip(quals * 400. / len(quals), 0, 1)
  audio_offsets = audio_times_full - video_times_full
  plt.switch_backend('Agg')
  plt.scatter(video_times_full / 60., audio_offsets, s=3, c=lcs_rgba, label='Matches')
  audio_offsets = audio_times - video_times
  def expand_limits(start, end, ratio=.01):
    average = (end + start) / 2.
    half_diff = (end - start) / 2.
    half_diff *= (1 + ratio)
    return (average - half_diff, average + half_diff)
  plt.xlim(expand_limits(*(0, np.max(video_times) / 60.)))
  plt.ylim(expand_limits(*(np.min(audio_offsets) - 10 * TIMESTEP_SIZE_SECONDS,
                           np.max(audio_offsets) + 10 * TIMESTEP_SIZE_SECONDS), .05))
  if stretch_audio:
    plt.plot(video_times / 60., audio_offsets, 'r-', lw=.5, label='Replaced Audio')
    audio_times_unreplaced = []
    video_times_unreplaced = []
    for i in range(len(video_times) - 1):
      slope = (audio_times[i+1] - audio_times[i]) / (video_times[i+1] - video_times[i])
      if abs(1 - slope) > MAX_RATE_RATIO_DIFF_ALIGN:
        video_times_unreplaced.extend(video_times[i:i+2])
        audio_times_unreplaced.extend(audio_times[i:i+2])
        video_times_unreplaced.append(video_times[i+1])
        audio_times_unreplaced.append(np.nan)
    if len(video_times_unreplaced) > 0:
      video_times_unreplaced = np.array(video_times_unreplaced)
      audio_times_unreplaced = np.array(audio_times_unreplaced)
      audio_offsets = audio_times_unreplaced - video_times_unreplaced
      plt.plot(video_times_unreplaced / 60., audio_offsets, 'c-', lw=1, label='Original Audio')
  else:
    plt.plot(video_times / 60., audio_offsets, 'r-', lw=1, label='Combined Media')
  plt.xlabel('Original Video Time (minutes)')
  plt.ylabel('Original Audio Description Offset (seconds behind video)')
  plt.title(f"Alignment - Media Similarity {similarity_percent:.2f}%")
  plt.legend().legend_handles[0].set_color(scatter_color)
  plt.tight_layout()
  plt.savefig(plot_filename_no_ext + '.png', dpi=400)
  plt.clf()
  with open(plot_filename_no_ext + '.txt', 'w') as file:
    parameters = {'stretch_audio':stretch_audio, 'no_pitch_correction':no_pitch_correction}
    print(f"Parameters: {parameters}", file=file)
    this_script_path = os.path.abspath(__file__)
    print(f"Version Hash: {get_version_hash(this_script_path)}", file=file)
    video_offset = video_times[0] - audio_times[0]
    print(f"Input file similarity: {similarity_percent:.2f}%", file=file)
    print("Main changes needed to video to align it to audio input:", file=file)
    print(f"Start Offset: {-video_offset:.2f} seconds", file=file)
    print(f"Median Rate Change: {(median_slope-1.)*100:.2f}%", file=file)
    for i in range(len(video_times) - 1):
      slope = (video_times[i+1] - video_times[i]) / (audio_times[i+1] - audio_times[i])
      def str_from_time(seconds):
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:2.0f}:{minutes:02.0f}:{seconds:06.3f}"
      print(f"Rate change of {(slope-1.)*100:8.1f}% from {str_from_time(video_times[i])} to " + \
            f"{str_from_time(video_times[i+1])} aligning with audio from " + \
            f"{str_from_time(audio_times[i])} to {str_from_time(audio_times[i+1])}", file=file)

# use the smooth alignment to replace runs of video sound with corresponding described audio
def replace_aligned_segments(video_arr, audio_desc_arr, audio_desc_times, video_times, no_pitch_correction):
  # perform quadratic interpolation of the audio description's waveform
  # this allows it to be stretched to match the corresponding video segment
  def audio_desc_arr_interp(samples):
    chunk_size = 10**5
    interpolated_chunks = []
    for chunk in (samples[i:i+chunk_size] for i in range(0, len(samples), chunk_size)):
      interp_bounds = (max(int(chunk[0]-2), 0),
                       min(int(chunk[-1]+2), audio_desc_arr.shape[1]))
      interp = scipy.interpolate.interp1d(np.arange(*interp_bounds),
                                          audio_desc_arr[:,slice(*interp_bounds)],
                                          copy=False, bounds_error=False, fill_value=0,
                                          kind='quadratic', assume_sorted=True)
      interpolated_chunks.append(interp(chunk).astype(np.float16))
    return np.hstack(interpolated_chunks)
  
  # yields matrices of pearson correlations indexed by the first window's start and
  # the second window's offset from the first window
  # the output matrix is truncated to the valid square with positive offsets
  # if negative=True, it is truncated to the valid square with negative offsets
  # subsequent yields are the adjacent square following the previously yielded one
  def get_pearson_corrs_generator(input, negative, jumps, window_size=512):
    # processing the entire vector at once is faster, but uses too much memory
    # instead, parse the input vector in pieces with a recursive call
    max_cached_chunks = 50
    cut = max_cached_chunks * window_size
    if input.shape[1] > (max_cached_chunks + 2) * 1.1 * window_size:
      is_first_iter = True
      while True:
        output_start = 0 if is_first_iter else 1
        is_last_iter = (input.shape[1] <= (max_cached_chunks + 2) * 1.1 * window_size)
        output_end = None if is_last_iter else max_cached_chunks
        input_end = None if is_last_iter else (cut + window_size)
        yield from itertools.islice(get_pearson_corrs_generator(input[:,:input_end], negative, jumps),
                                    output_start, output_end)
        if is_last_iter:
          return
        input = input[:,cut-window_size:]
        is_first_iter = False
    if input.shape[1] < 3 * window_size - 1:
      raise RuntimeError("Invalid state in Pearson generator.")
    pearson_corrs = np.zeros((len(jumps), input.shape[1] - window_size + 1)) - np.inf
    # calculate dot products of pairs of windows (i.e. autocorrelation)
    # avoids redundant calculations by substituting differences in the cumulative sum of products
    self_corr = np.sum(input.astype(np.float32)**2, axis=0)
    corr_cumsum = np.cumsum(self_corr, dtype=np.float64)
    corr_cumsum[window_size:] -= corr_cumsum[:-window_size]
    window_rms = corr_cumsum[window_size-1:]
    epsilon = 1e-4 * max(1, np.max(window_rms))
    window_rms = np.sqrt(window_rms + epsilon)
    for jump_index, jump in enumerate(jumps):
      autocorrelation = np.sum(input[:,jump:].astype(np.float32) * input[:,:input.shape[1]-jump], axis=0)
      autocorr_cumsum = np.cumsum(autocorrelation, dtype=np.float64)
      autocorr_cumsum[window_size:] -= autocorr_cumsum[:-window_size]
      if negative:
        pearson_corrs[jump_index, jump:] = autocorr_cumsum[window_size-1:] + epsilon
        pearson_corrs[jump_index, jump:] /= window_rms[:len(window_rms)-jump]
      else:
        pearson_corrs[jump_index, :pearson_corrs.shape[1]-jump] = autocorr_cumsum[window_size-1:] + epsilon
        pearson_corrs[jump_index, :pearson_corrs.shape[1]-jump] /= window_rms[jump:]
    # divide by RMS of constituent windows to get Pearson correlations
    pearson_corrs = pearson_corrs / window_rms[None,:]
    pearson_corrs = pearson_corrs.T
    for chunk_index in range(0, input.shape[1] // window_size):
      yield pearson_corrs[chunk_index*window_size:(chunk_index+1)*window_size]
  
  def stretch(input, output, window_size=512, max_drift=512*3):
    drift_window_size = max_drift * 2 + 1
    num_input_samples = input.shape[1]
    num_output_samples = output.shape[1]
    total_offset_samples = num_output_samples - num_input_samples
    jumps = [506, 451, 284, 410, 480, 379, 308, 430, 265, 494]
    # use all jumps when given unreachable or difficult to reach offsets (i.e. Frobenius coin problem)
    # otherwise, skip most jumps to trade off a little performance for a lot of speed
    if abs(total_offset_samples) < 10000:
      if abs(total_offset_samples) > 1000:
        jumps.extend([MIN_STRETCH_OFFSET + offset for offset in (2**np.arange(8))-1])
      else:
        jumps = range(MIN_STRETCH_OFFSET, window_size)
    num_windows = (num_input_samples // window_size)
    window_to_offset = lambda window_index: (total_offset_samples * \
                                             min((num_windows - 1), max(0, window_index))) // (num_windows - 1)
    # note the absolute value in the drift
    # the following calculations also use the absolute value of the jumps
    # their signs flip together, so this saves on casework in the code
    # after the optimal route is determined, the sign of the jumps will be reintroduced
    window_to_offset_diff = lambda window_index: abs(window_to_offset(window_index) - \
                                                     window_to_offset(window_index - 1))
    backpointers = np.zeros((num_windows, drift_window_size), dtype=np.int16)
    best_jump_locations = np.zeros((num_windows, len(jumps)), dtype=np.int16)
    cum_loss = np.zeros((3, drift_window_size)) + np.inf
    cum_loss[1:, max_drift] = 0
    last_offset_diff = 0
    # if the output needs to be longer than the input, we need to jump backwards in the input
    pearson_corrs_generator = get_pearson_corrs_generator(input, (total_offset_samples > 0), jumps)
    for window_index in range(num_windows):
      corrs = next(pearson_corrs_generator)
      # for each jump distance, determine the best input index in the window to make that jump
      best_jump_locations[window_index] = np.argmax(corrs, axis=0)
      best_jump_losses = 1 - corrs[best_jump_locations[window_index], np.arange(corrs.shape[1])]
      offset_diff = window_to_offset_diff(window_index)
      offset_diff2 = offset_diff + last_offset_diff
      offset_jump_losses = np.zeros((len(jumps)+1, drift_window_size)) + np.inf
      # consider not jumping at all, copying the loss from the corresponding offset one window back
      offset_jump_slice = slice(None, offset_jump_losses.shape[1] - offset_diff)
      offset_jump_losses[0,offset_jump_slice] = cum_loss[(window_index-1)%3,offset_diff:]
      for jump_index, jump in enumerate(jumps):
        truncation_amount = offset_diff2 - jump
        offset_jump_slice = slice(jump, drift_window_size - max(0, truncation_amount))
        cum_loss_slice = slice(offset_diff2, drift_window_size + min(0, truncation_amount))
        # consider jumping the given distance from two windows back
        # a window is skipped when jumping to prevent overlapping crossfades
        offset_jump_losses[jump_index+1, offset_jump_slice] = cum_loss[(window_index-2)%3, cum_loss_slice] + \
                                                              best_jump_losses[jump_index]
      best_jumps = np.argmin(offset_jump_losses, axis=0)
      backpointers[window_index] = best_jumps
      cum_loss[window_index%3] = offset_jump_losses[best_jumps, np.arange(offset_jump_losses.shape[1])]
      last_offset_diff = offset_diff
    drift = max_drift
    best_jumps = []
    skip_window = False
    for window_index in range(num_windows - 1, -1, -1):
      drift += window_to_offset_diff(window_index + 1)
      if skip_window:
        skip_window = False
        continue
      best_jump_index = backpointers[window_index, drift] - 1
      if best_jump_index == -1:
        continue
      best_jump = jumps[best_jump_index]
      jump_input_index = window_index * window_size + \
                         best_jump_locations[window_index, best_jump_index].item()
      drift -= best_jump
      skip_window = True
      best_jumps.append((jump_input_index, best_jump))
    best_jumps = best_jumps[::-1]
    best_jumps = np.array(best_jumps)
    # reintroduce the sign of the jump distances
    # if the output is longer, use backwards jumps in the input to duplicate samples
    # if the output is shorter, use forwards jumps in the input to remove samples
    if total_offset_samples > 0:
      best_jumps[:,1] *= -1
    jump_input_indices = best_jumps[:,0]
    jump_distances = best_jumps[:,1]
    # calculate starts and ends of segments that will be copied from input to output
    input_starts = np.concatenate(([0], jump_input_indices + jump_distances))
    input_ends = np.concatenate((jump_input_indices, [input.shape[1]]))
    chunk_lengths = input_ends - input_starts
    output_ends = np.cumsum(chunk_lengths)
    output_starts = np.concatenate(([0], output_ends[:-1]))
    bump = scipy.signal.windows.hann(2 * window_size + 1)
    bump_head = bump[:window_size]
    bump_tail = bump[window_size:-1]
    output[:,:window_size] = input[:,:window_size]
    for in_start, in_end, out_start, out_end in zip(input_starts, input_ends, output_starts, output_ends):
      output[:,out_start:out_start+window_size] *= bump_tail
      output[:,out_start:out_start+window_size] += input[:,in_start:in_start+window_size] * bump_head
      output[:,out_start+window_size:out_end+window_size] = input[:,in_start+window_size:in_end+window_size]
  
  x = audio_desc_times
  y = video_times
  x_samples = (x * AUDIO_SAMPLE_RATE).astype(int)
  y_samples = (y * AUDIO_SAMPLE_RATE).astype(int)
  diff_x_samples = np.diff(x_samples)
  diff_y_samples = np.diff(y_samples)
  slopes = diff_x_samples / diff_y_samples
  total_offset_samples = diff_y_samples - diff_x_samples
  y_midpoint_samples = (y_samples[:-1] + y_samples[1:]) // 2
  progress_update_interval = (video_arr.shape[1] // 100) + 1
  last_progress_update = -1
  for i in range(len(x) - 1):
    if diff_y_samples[i] < (MIN_DURATION_TO_REPLACE_SECONDS * AUDIO_SAMPLE_RATE) or \
       np.abs(1 - slopes[i]) > MAX_RATE_RATIO_DIFF_ALIGN:
      continue
    video_arr_slice = video_arr[:,slice(*y_samples[i:i+2])]
    progress = int(y_midpoint_samples[i] // progress_update_interval)
    if progress > last_progress_update:
      last_progress_update = progress
      print(f"  stretching audio:{progress:3d}%                        \r", end='')
    # only apply pitch correction if the difference would be noticeable
    if no_pitch_correction or np.abs(1 - slopes[i]) <= JUST_NOTICEABLE_DIFF_IN_FREQ_RATIO or \
       abs(total_offset_samples[i]) < MIN_STRETCH_OFFSET:
      # construct a stretched audio description waveform using the quadratic interpolator
      sample_points = np.linspace(*x_samples[i:i+2], num=diff_y_samples[i], endpoint=False)
      video_arr_slice[:] = audio_desc_arr_interp(sample_points)
    else:
      stretch(audio_desc_arr[:,slice(*x_samples[i:i+2])], video_arr_slice)

# Convert piece-wise linear fit to ffmpeg expression for editing video frame timestamps
def encode_fit_as_ffmpeg_expr(audio_desc_times, video_times, video_offset):
  # PTS is the input frame's presentation timestamp, which is when frames are displayed
  # TB is the timebase, which is how many seconds each unit of PTS corresponds to
  # the output value of the expression will be the frame's new PTS
  setts_cmd = ['TS']
  # each segment of the linear fit can be encoded as a single clip function
  setts_cmd.append('+(0')
  x = audio_desc_times
  y = video_times
  diff_x = np.diff(x)
  diff_y = np.diff(y)
  slopes = diff_x / diff_y
  for i in range(len(audio_desc_times) - 1):
    setts_cmd.append(f'+clip(TS-{y[i]-video_offset:.4f}/TB,0,{max(0,diff_y[i]):.4f}/TB)*{slopes[i]-1:.9f}')
  setts_cmd.append(')')
  setts_cmd = ''.join(setts_cmd)
  return setts_cmd

def get_ffmpeg():
  return static_ffmpeg.run._get_or_fetch_platform_executables_else_raise_no_lock()[0]

def get_ffprobe():
  return static_ffmpeg.run._get_or_fetch_platform_executables_else_raise_no_lock()[1]

def get_key_frame_data(video_file, time=None, entry='pts_time'):
  interval = f'%+{max(60,time+40)}' if time != None else '%'
  key_frames = ffmpeg.probe(video_file, cmd=get_ffprobe(), select_streams='V', show_frames=None, 
                            skip_frame='nokey', read_intervals=interval,
                            show_entries='frame='+entry)['frames']
  return np.array([float(frame[entry]) for frame in key_frames if entry in frame])

# finds the average timestamp of (i.e. midpoint between) the key frames on either side of input time
def get_closest_key_frame_time(video_file, time):
  key_frame_times = get_key_frame_data(video_file, time)
  key_frame_times = key_frame_times if len(key_frame_times) > 0 else np.array([0])
  prev_key_frame_times = key_frame_times[key_frame_times <= time]
  prev_key_frame = np.max(prev_key_frame_times) if len(prev_key_frame_times) > 0 else time
  next_key_frame_times = key_frame_times[key_frame_times > time]
  next_key_frame = np.min(next_key_frame_times) if len(next_key_frame_times) > 0 else time
  return (prev_key_frame + next_key_frame) / 2.

# outputs a new media file with the replaced audio (which includes audio descriptions)
def write_replaced_media_to_disk(output_filename, media_arr, video_file=None, audio_desc_file=None,
                                 setts_cmd=None, video_offset=None, after_start_key_frame=None):
  # if a media array is given, stretch_audio is enabled and media_arr should be added to the video
  if media_arr is not None:
    media_input = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le', ac=2, ar=AUDIO_SAMPLE_RATE)
    # if no video file is given, the input "video" was an audio file and the output should be too
    if video_file is None:
      write_command = ffmpeg.output(media_input, output_filename, loglevel='error').overwrite_output()
    else:
      original_video = ffmpeg.input(video_file, dn=None)
      # "-max_interleave_delta 0" is sometimes necessary to fix an .mkv bug that freezes audio/video:
      #   ffmpeg bug warning: [matroska @ 0000000002c814c0] Starting new cluster due to timestamp
      # more info about the bug and fix: https://reddit.com/r/ffmpeg/comments/efddfs/
      write_command = ffmpeg.output(media_input, original_video, output_filename,
                                    acodec='copy', vcodec='copy', scodec='copy',
                                    max_interleave_delta='0', loglevel='error',
                                    **{"c:a:0": "aac", "disposition:a:1": "original",
                                       "metadata:s:a:1": "title=original",
                                       "disposition:a:0": "default+visual_impaired+descriptions",
                                       "metadata:s:a:0": "title=AD"}).overwrite_output()
    run_async_ffmpeg_command(write_command, media_arr, f"write output file: {output_filename}")
  else:
    start_offset = video_offset - after_start_key_frame
    media_input = ffmpeg.input(audio_desc_file, itsoffset=f'{max(0, start_offset):.6f}')
    original_video = ffmpeg.input(video_file, an=None, ss=f'{after_start_key_frame:.6f}',
                                  itsoffset=f'{max(0, -start_offset):.6f}', dn=None)
    # wav files don't have codecs compatible with most video containers, so we convert to aac
    audio_codec = 'copy' if os.path.splitext(audio_desc_file)[1] != '.wav' else 'aac'
    # flac audio may only have experimental support in some video containers (e.g. mp4)
    standards = 'normal' if os.path.splitext(audio_desc_file)[1] != '.flac' else 'experimental'
    # add frag_keyframe flag to prevent some players from ignoring audio/video start offsets
    # set both pts and dts simultaneously in video manually, as ts= does not do the same thing
    write_command = ffmpeg.output(media_input, original_video, output_filename,
                                  acodec=audio_codec, vcodec='copy', scodec='copy',
                                  max_interleave_delta='0', loglevel='error',
                                  strict=standards, movflags='frag_keyframe',
                                  **{'bsf:v': f'setts=pts=\'{setts_cmd}\':dts=\'{setts_cmd}\'',
                                     'bsf:s': f'setts=ts=\'{setts_cmd}\'',
                                     "disposition:a:0": "default+visual_impaired+descriptions",
                                     "metadata:s:a:0": "title=AD"}).overwrite_output()
    run_ffmpeg_command(write_command, f"write output file: {output_filename}")

# check whether static_ffmpeg has already installed ffmpeg and ffprobe
def is_ffmpeg_installed():
  ffmpeg_dir = static_ffmpeg.run.get_platform_dir()
  indicator_file = os.path.join(ffmpeg_dir, "installed.crumb")
  return os.path.exists(indicator_file)

def get_energy(arr):
  # downsample of 105, hann size 15, downsample by 2 gives 210 samples per second, ~65 halfwindows/second
  decimation = 105
  decimation2 = 2
  arr_clip = arr[:,:(arr.shape[1] - (arr.shape[1] % decimation))].reshape(arr.shape[0], -1, decimation)
  energy = np.einsum('ijk,ijk->j', arr_clip, arr_clip, dtype=np.float32) / (decimation * arr.shape[0])
  hann_window = scipy.signal.windows.hann(15)[1:-1].astype(np.float32)
  hann_window /= np.sum(hann_window)
  energy_smooth = np.convolve(energy, hann_window, mode='same')
  energy_smooth = np.log10(1 + energy_smooth) / 2.
  return energy_smooth[::decimation2]

def get_zero_crossings(arr):
  xings = np.diff(np.signbit(arr), prepend=False, axis=-1)
  xings_clip = xings[:,:(xings.shape[1] - (xings.shape[1] % 210))].reshape(xings.shape[0], -1, 210)
  zero_crossings = np.sum(np.abs(xings_clip), axis=(0,2)).astype(np.float32)
  if xings.shape[0] == 1:
    zero_crossings *= 2
  hann_window = scipy.signal.windows.hann(15)[1:-1].astype(np.float32)
  hann_window = hann_window / np.sum(hann_window)
  zero_crossings_smooth = np.convolve(zero_crossings, hann_window, mode='same')
  return zero_crossings_smooth

def downsample_blur(arr, downsample, blur):
  hann_window = scipy.signal.windows.hann(downsample*blur+2)[1:-1].astype(np.float32)
  hann_window = hann_window / np.sum(hann_window)
  arr = arr[:len(arr)-(len(arr)%downsample)]
  return sum((np.convolve(arr[i::downsample], hann_window[i::downsample],
                          mode='same') for i in range(downsample)))

def get_freq_bands(arr):
  arr = np.mean(arr, axis=0) if arr.shape[0] > 1 else arr[0]
  arr = arr[:len(arr)-(len(arr)%210)]
  downsamples = [5, 7, 6]
  decimation = 1
  freq_bands = []
  for downsample in downsamples:
    if downsample == downsamples[-1]:
      band_bottom = np.array(0).reshape(1)
    else:
      band_bottom = downsample_blur(arr, downsample, 3)
    decimation *= downsample
    arr = arr.reshape(-1, downsample)
    band_energy = sum(((arr[:,i] - band_bottom) ** 2 for i in range(downsample)))
    freq_band = downsample_blur(band_energy, (210 // decimation), 15) / 210
    freq_band = np.log10(1 + freq_band) / 2.
    freq_bands.append(freq_band)
    arr = band_bottom
  return freq_bands

def align(video_features, audio_desc_features, video_energy, audio_desc_energy):
  samples_per_node = 210 // TIMESTEPS_PER_SECOND
  hann_window_unnormed = scipy.signal.windows.hann(2*samples_per_node+1)[1:-1]
  hann_window = hann_window_unnormed / np.sum(hann_window_unnormed)
  get_mean = lambda arr: np.convolve(hann_window, arr, mode='same')
  get_uniform_norm = lambda arr: np.convolve(np.ones(hann_window.shape), arr ** 2, mode='valid') ** .5
  def get_uniform_norms(features):
    return [np.clip(get_uniform_norm(feature), .001, None) for feature in features]
  
  print("  memorizing video...        \r", end='')
  video_features_mean_sub = [feature - get_mean(feature) for feature in video_features]
  audio_desc_features_mean_sub = [feature - get_mean(feature) for feature in audio_desc_features]
  video_uniform_norms = get_uniform_norms(video_features_mean_sub)
  audio_desc_uniform_norms = get_uniform_norms(audio_desc_features_mean_sub)
  
  num_bins = 7
  bin_spacing = 6
  bins_width = (num_bins - 1) * bin_spacing + 1
  bins_start = samples_per_node - 1 - (bins_width // 2)
  bins_end = bins_start + bins_width
  video_dicts = [defaultdict(set) for feature in video_features_mean_sub]
  edges = np.array(np.meshgrid(*([np.arange(2)]*num_bins), indexing='ij')).reshape(num_bins,-1).T
  bin_offsets = []
  for edge in edges:
    bin_offset = np.array(np.meshgrid(*[np.arange(x+1) for x in edge], indexing='ij'))
    bin_offsets.append(np.dot(bin_offset.reshape(num_bins,-1)[::-1].T, 7**np.arange(num_bins)))
  
  for video_dict, feature, norm in zip(video_dicts, video_features_mean_sub, video_uniform_norms):
    bins = np.hstack([feature[bins_start+i:-bins_end+i+1, None] for i in bin_spacing * np.arange(num_bins)])
    bins /= norm[:,None]
    bins = 8 * bins + 3.3
    np.clip(bins, 0, 6, out=bins)
    bin_offset_indices = np.dot(((bins % 1) > .6), 2**np.arange(num_bins))
    bins = np.dot(np.floor(bins).astype(int), 7**np.arange(num_bins)).tolist()
    not_quiet = (video_energy[:-len(hann_window)] > .5)
    for i in np.arange(len(video_energy) - len(hann_window))[not_quiet].tolist()[::4]:
      bin = bins[i]
      for bin_offset in bin_offsets[bin_offset_indices[i]].tolist():
        video_dict[bin + bin_offset].add(i)
  
  print("  matching audio...  \r", end='')
  audio_desc_bins = []
  audio_desc_bin_offset_indices = []
  for feature, norm in zip(audio_desc_features_mean_sub, audio_desc_uniform_norms):
    bins = np.hstack([feature[bins_start+i:-bins_end+i+1, None] for i in bin_spacing * np.arange(num_bins)])
    bins /= norm[:,None]
    bins = 8 * bins + 3.5
    bins = np.floor(bins).astype(int)
    np.clip(bins, 0, 6, out=bins)
    audio_desc_bins.append(np.dot(bins, 7**np.arange(num_bins)).tolist())
  
  def pairwise_intersection(set1, set2, set3):
    return (set1 & set2).union((set1 & set3), (set2 & set3))
  def triwise_intersection(set1, set2, set3, set4, set5):
    set123 = pairwise_intersection(set1, set2, set3)
    return (set123 & set4) | (set123 & set5)
  best_so_far = SortedList(key=lambda x:x[0])
  best_so_far.add((-1,-1,0))
  backpointers = {}
  not_quiet = (audio_desc_energy[:-len(hann_window)] > .5)
  for i in np.arange(len(audio_desc_energy) - len(hann_window))[not_quiet].tolist():
    match_sets = [video_dict[bins[i]] for bins, video_dict in zip(audio_desc_bins, video_dicts)]
    common = triwise_intersection(*match_sets)
    match_points = []
    for video_index in common:
      prob = 1
      for j in range(3):
        corr = np.dot(audio_desc_features_mean_sub[j][i:i+2*samples_per_node-1],
                      video_features_mean_sub[j][video_index:video_index+2*samples_per_node-1])
        corr /= audio_desc_uniform_norms[j][i] * video_uniform_norms[j][video_index]
        prob *= max(1e-8, (1 - corr))  # Naive Bayes probability
      prob = prob ** 2.9  # empirically determined, ranges from 2.5-3.4
      if prob > 1e-8:
        continue
      qual = min(50, (prob / 1e-12) ** (-1. / 3))  # remove Naive Bayes assumption
      match_points.append((video_index, qual))
    audio_desc_index = i
    for video_index, qual in sorted(match_points):
      cur_index = best_so_far.bisect_right((video_index,))
      prev_video_index, prev_audio_desc_index, prev_cum_qual = best_so_far[cur_index-1]
      cum_qual = prev_cum_qual + qual
      while (cur_index < len(best_so_far)) and (best_so_far[cur_index][2] <= cum_qual):
        del best_so_far[cur_index]
      best_so_far.add((video_index, audio_desc_index, cum_qual))
      backpointers[(video_index, audio_desc_index)] = (prev_video_index, prev_audio_desc_index)
  del video_dicts
  path = [best_so_far[-1][:2]]
  while path[-1][:2] in backpointers:
    # failsafe to prevent an infinite loop that should never happen anyways
    if len(path) > 10**8:
      raise RuntimeError("Infinite Loop Encountered!")
    path.append(backpointers[path[-1][:2]])
  path.pop()
  path.reverse()
  if len(path) < max(min(len(video_energy), len(audio_desc_energy)) / 500., 5 * 210):
    raise RuntimeError("Alignment failed, are the input files mismatched?")
  y, x = np.array(path).T
  
  half_hann_window = hann_window[:samples_per_node-1] / np.sum(hann_window[:samples_per_node-1])
  half_samples_per_node = samples_per_node // 2
  fit_delay = samples_per_node + half_samples_per_node - 2
  diff_by = lambda arr, offset=half_samples_per_node: arr[offset:] - arr[:-offset]
  def get_continuity_err(x, y, deriv=False):
    x_smooth_future = np.convolve(x, half_hann_window, mode='valid')
    y_smooth_future = np.convolve(y, half_hann_window, mode='valid')
    slopes_future = diff_by(y_smooth_future) / diff_by(x_smooth_future)
    offsets_future = y_smooth_future[:-half_samples_per_node] - \
                     x_smooth_future[:-half_samples_per_node] * slopes_future
    x_smooth_past = np.convolve(x, half_hann_window[::-1], mode='valid')
    y_smooth_past = np.convolve(y, half_hann_window[::-1], mode='valid')
    slopes_past = diff_by(y_smooth_past) / diff_by(x_smooth_past)
    offsets_past = y_smooth_past[half_samples_per_node:] - \
                   x_smooth_past[half_samples_per_node:] * slopes_past
    continuity_err = np.full(len(x) - (1 if deriv else 0), np.inf)
    fit_delay_offset = fit_delay - (1 if deriv else 0)
    continuity_err[:-fit_delay_offset] = np.abs(slopes_future * x[:-fit_delay] + \
                                                offsets_future - y[:-fit_delay])
    continuity_err[fit_delay_offset:] = np.minimum(continuity_err[fit_delay_offset:],
                                                   np.abs(slopes_past * x[fit_delay:] + \
                                                          offsets_past - y[fit_delay:]))
    return continuity_err
  
  print("  refining match: pass 1 of 2...\r", end='')
  continuity_err = get_continuity_err(x, y)
  errs = (continuity_err < 3)
  x = x[errs]
  y = y[errs]
  
  audio_desc_features_scaled = []
  video_features_scaled = []
  for video_feature, audio_desc_feature in zip(video_features, audio_desc_features):
    audio_desc_feature_std = np.std(audio_desc_feature)
    scale_factor = np.linalg.lstsq(video_feature[y][:,None], audio_desc_feature[x], rcond=None)[0]
    audio_desc_features_scaled.append(audio_desc_feature / audio_desc_feature_std)
    video_features_scaled.append(video_feature * scale_factor / audio_desc_feature_std)
  audio_desc_features_scaled = np.array(list(zip(*(audio_desc_features_scaled[:3]))))
  video_features_scaled = np.array(list(zip(*(video_features_scaled[:3]))))
  
  smooth_x = get_mean(x)
  smooth_y = get_mean(y)
  slopes = np.diff(smooth_y) / np.diff(smooth_x)
  offsets = smooth_y[:-1] - smooth_x[:-1] * slopes
  err_y = slopes * x[:-1] + offsets - y[:-1]
  compressed_x, compressed_y = [], []
  def extend_all(index, compress=False, num=70):
    compressed_x.extend([np.mean(x[index:index+num])] if compress else x[index:index+num])
    compressed_y.extend([np.mean(y[index:index+num])] if compress else y[index:index+num])
  extend_all(0, num=10)
  for i in range(10, len(x) - 80, 70):
    extend_all(i, compress=np.all(np.abs(err_y[i:i+70]) < 3))
  extend_all(i+70)
  
  x = compressed_x
  y = compressed_y
  
  match_dict = defaultdict(list)
  x_unique = [-1]
  for audio_desc_index, video_index in zip(x, y):
    match_dict[audio_desc_index].append(video_index)
    if audio_desc_index != x_unique[-1]:
      x_unique.append(audio_desc_index)
  x = np.array(x_unique[1:])
  y = np.array([np.mean(match_dict[audio_desc_index]) for audio_desc_index in x])
  
  # L1-Minimization to solve the alignment problem using a linear program
  # the absolute value functions needed for "absolute error" can be represented
  # in a linear program by splitting variables into positive and negative pieces
  # and constraining each to be positive (done by default in scipy's linprog)
  num_fit_points = len(x)
  x_diffs = np.diff(x)
  y_diffs = np.diff(y)
  jump_cost_base = 10.
  jump_costs = np.full(num_fit_points - 1, jump_cost_base)
  continuity_err = get_continuity_err(x, y, deriv=True)
  jump_costs /= np.maximum(1, np.sqrt(continuity_err / 3.))
  rate_change_jump_costs = np.full(num_fit_points - 1, .001)
  rate_change_costs = np.full(num_fit_points - 2, jump_cost_base * 4000)
  shot_noise_costs = np.full(num_fit_points, .01)
  shot_noise_jump_costs = np.full(num_fit_points - 1, 3)
  shot_noise_bound = 2.
  c = np.hstack([np.ones(2 * num_fit_points),
                 jump_costs,
                 jump_costs,
                 shot_noise_costs,
                 shot_noise_costs,
                 shot_noise_jump_costs,
                 shot_noise_jump_costs,
                 rate_change_jump_costs,
                 rate_change_jump_costs,
                 rate_change_costs,
                 rate_change_costs,
                 [0,]])
  fit_err_coeffs = scipy.sparse.diags([-1. / x_diffs,
                                        1. / x_diffs],
                                      offsets=[0,1],
                                      shape=(num_fit_points - 1, num_fit_points)).tocsc()
  jump_coeffs = scipy.sparse.diags([ 1. / x_diffs],
                                   offsets=[0],
                                   shape=(num_fit_points - 1, num_fit_points - 1)).tocsc()
  A_eq1 = scipy.sparse.hstack([ fit_err_coeffs,
                               -fit_err_coeffs,
                                jump_coeffs,
                               -jump_coeffs,
                                scipy.sparse.csc_matrix((num_fit_points - 1, 2 * num_fit_points)),
                                jump_coeffs,
                               -jump_coeffs,
                                jump_coeffs,
                               -jump_coeffs,
                                scipy.sparse.csc_matrix((num_fit_points - 1, 2 * num_fit_points - 4)),
                                np.ones((num_fit_points - 1, 1))])
  A_eq2 = scipy.sparse.hstack([ scipy.sparse.csc_matrix((num_fit_points - 1, 4 * num_fit_points - 2)),
                                scipy.sparse.diags([-1., 1.], offsets=[0, 1],
                                                   shape=(num_fit_points - 1, num_fit_points)).tocsc(),
                                scipy.sparse.diags([1., -1.], offsets=[0, 1],
                                                   shape=(num_fit_points - 1, num_fit_points)).tocsc(),
                               -scipy.sparse.eye(num_fit_points - 1),
                                scipy.sparse.eye(num_fit_points - 1),
                                scipy.sparse.csc_matrix((num_fit_points - 1, 4 * num_fit_points - 6)),
                                scipy.sparse.csc_matrix((num_fit_points - 1, 1))])
  slope_change_coeffs = scipy.sparse.diags([-1. / x_diffs[:-1],
                                             1. / x_diffs[1:]],
                                           offsets=[0,1],
                                           shape=(num_fit_points - 2, num_fit_points - 1)).tocsc()
  A_eq3 = scipy.sparse.hstack([scipy.sparse.csc_matrix((num_fit_points - 2, 8 * num_fit_points - 4)),
                               slope_change_coeffs,
                               -slope_change_coeffs,
                               -scipy.sparse.eye(num_fit_points - 2),
                               scipy.sparse.eye(num_fit_points - 2),
                               scipy.sparse.csc_matrix((num_fit_points - 2, 1))])
  A_eq = scipy.sparse.vstack([A_eq1, A_eq2, A_eq3])
  b_eq = y_diffs / x_diffs
  b_eq = np.hstack((b_eq, np.zeros(2 * num_fit_points - 3)))
  bounds = [[0, None]] * (4 * num_fit_points - 2) + \
           [[0, shot_noise_bound]] * (2 * num_fit_points) + \
           [[0, None]] * (6 * num_fit_points - 8) + \
           [[None, None]]
  fit = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ds')
  # if dual simplex solver encounters numerical problems, retry with interior point solver
  if not fit.success and fit.status == 4:
    fit = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs-ipm')
  if not fit.success:
    print(fit)
    raise RuntimeError("Smooth Alignment L1-Min Optimization Failed!")
  
  # combine positive and negative components of variables
  fit_err = fit.x[                  :  num_fit_points  ] - \
            fit.x[  num_fit_points  :2*num_fit_points  ]
  slope_jumps = fit.x[8*num_fit_points-4: 9*num_fit_points-5] - \
                fit.x[9*num_fit_points-5:10*num_fit_points-6]
  median_slope = fit.x[-1]
  slopes = median_slope + (slope_jumps / x_diffs)
  
  # subtract fit errors from nodes to retrieve the smooth fit's coordinates
  smooth_path = [(x, y) for x,y in zip(x, y - fit_err)]
  
  print("  refining match: pass 2 of 2...\r", end='')
  slopes_plus_ends = np.hstack((slopes[:1], slopes, slopes[-1:]))
  extensions = []
  extend_radius = 210 * 30  # +/- 30 seconds
  video_interp = scipy.interpolate.make_interp_spline(np.arange(len(video_features_scaled)),
                                                      video_features_scaled, k=1)
  colinear_dict = defaultdict(list)
  for i, (x, y) in enumerate(smooth_path):
    for slope in slopes_plus_ends[i:i+2]:
      if (slope < .1) or (slope > 10):
        continue
      offset = y - slope * x
      colinear_dict[(round(slope, 6), int(round(offset, 0)))].append((x, y))
  line_clusters = []
  added_keys = set()
  for (slope, offset), indices in sorted(colinear_dict.items(), key=lambda x: -len(x[1])):
    if (slope, offset) in added_keys:
      continue
    line_clusters.append(indices)
    added_keys.add((slope, offset))
    del colinear_dict[(slope, offset)]
    for (slope2, offset2), indices2 in list(colinear_dict.items()):
      if (abs(indices2[ 0][1] - (indices2[ 0][0] * slope + offset)) < 3) and \
         (abs(indices2[-1][1] - (indices2[-1][0] * slope + offset)) < 3):
        line_clusters[-1].extend(colinear_dict[(slope2, offset2)])
        added_keys.add((slope2, offset2))
        del colinear_dict[(slope2, offset2)]
  line_clusters = [sorted(cluster) for cluster in line_clusters]
  line_clusters = [x for x in line_clusters if (abs(x[0][0] - x[-1][0]) > 10) and len(x) > 5]
  
  for i, cluster in enumerate(line_clusters):
    x, y = np.array(cluster).T
    linear_fit = np.linalg.lstsq(np.hstack((np.ones((len(x), 1)), x[:, None])), y, rcond=None)[0]
    line_clusters[i] = (x, linear_fit[0], linear_fit[1])
  
  def get_x_limits(x, offset, slope, extend_horiz=extend_radius, buffer_vert=4):
    limits = (max(int(x[0])  - extend_horiz, 0),
              min(int(x[-1]) + extend_horiz, len(audio_desc_features_scaled) - 1))
    limits = (max(limits[0], int(np.ceil((buffer_vert - offset) / slope))),
              min(limits[1], int(np.floor((len(video_features_scaled) - buffer_vert - offset) / slope))))
    return limits
  def get_audio_video_matches(limits, slope, offset):
    x = np.arange(*limits)
    y = slope * x + offset
    audio_match = audio_desc_features_scaled[slice(*limits)]
    video_match = video_interp(y)
    return x, y, audio_match, video_match
  
  audio_desc_max_energy = np.max(audio_desc_features_scaled[:,0])
  video_max_energy = np.max(video_features_scaled[:,0])
  points = [[] for i in range(len(audio_desc_features_scaled))]
  seen_points = set()
  for cluster_index, (x, offset, slope) in enumerate(line_clusters):
    limits = get_x_limits(x, offset, slope, extend_horiz=0)
    if limits[1] < limits[0] + 5:
      continue
    if limits[1] > limits[0] + 100:
      x, y, audio_match, video_match = get_audio_video_matches(limits, slope, offset)
      video_match_err = audio_match[1:-1] - video_match[1:-1]
      valid_matches = np.mean(video_match_err, axis=-1) < 0.1
      if np.count_nonzero(valid_matches) > 50:
        video_match_diff = (video_match[2:] - video_match[:-2]) / 2.
        video_match_err = video_match_err[valid_matches]
        video_match_diff = video_match_diff[valid_matches]
        x_valid = x[1:-1][valid_matches][:,None]
        A = video_match_diff.reshape(-1,1)
        linear_fit, residual, _, _ = np.linalg.lstsq(A, video_match_err.flat, rcond=None)
        explained_err_ratio = 1 - (residual / np.sum(video_match_err ** 2))
        stds_above_noise_mean = np.sqrt(explained_err_ratio * np.prod(video_match_err.shape)) - 1.
        if stds_above_noise_mean > 8 and abs(linear_fit[0]) < 2:
          offset += linear_fit[0]
    limits = get_x_limits(x, offset, slope)
    x, y, audio_match, video_match = get_audio_video_matches(limits, slope, offset)
    quals = np.sum(-.5 - np.log10(1e-4 + np.abs(audio_match - video_match)), axis=1)
    quals *= np.clip(video_match[:,0] + 2.5 - video_max_energy, 0, 1)
    quals += np.clip(audio_match[:,0] + 2.5 - audio_desc_max_energy, 0, 1) * .1
    energy_diffs = audio_match[:,0] - video_match[:,0]
    for i, j, qual in zip(x.tolist(), y.tolist(), quals.tolist()):
      point = (i, int(j))
      if point not in seen_points:
        seen_points.add(point)
        points[i].append((j, cluster_index, qual))
  del seen_points
  points = [sorted(point) for point in points]
  
  best_so_far = SortedList(key=lambda x:x[0])
  best_so_far.add((0, 0, -1, 0, 0))  # video_index, audio_desc_index, cluster_index, qual, cum_qual
  clusters_best_so_far = [(0, 0, 0, -1000) for cluster in line_clusters]
  backpointers = {}
  prev_cache = np.full((len(video_features_scaled), 5), -np.inf)
  prev_cache[0] = (0, 0, -1, 0, 0)  # video_index, audio_desc_index, cluster_index, qual, cum_qual
  reversed_min_points = [min(x)[0] if len(x) > 0 else np.inf for x in points[::-1]]
  forward_min = list(itertools.accumulate(reversed_min_points, min))[::-1]
  for i in range(len(audio_desc_features_scaled)):
    for j, cluster_index, qual in points[i]:
      cur_index = best_so_far.bisect_right((j,))
      prev_j, prev_i, prev_cluster_index, prev_qual, best_prev_cum_qual = best_so_far[cur_index-1]
      cluster_last = clusters_best_so_far[cluster_index]
      if cluster_last[3] >= best_prev_cum_qual:
        prev_j, prev_i, prev_qual, best_prev_cum_qual = cluster_last
        prev_cluster_index = cluster_index
      for prev_j_temp in range(max(0, int(j) - 2), int(j) + 1):
        prev_node = prev_cache[prev_j_temp].tolist()
        if cluster_index != prev_node[2]:
          prev_node[4] -= 100 + 100 * ((j - prev_node[0]) - (i - prev_node[1])) ** 2
        if prev_node[1] >= (i - 2) and \
           prev_node[0] <= j and \
           prev_node[4] >= best_prev_cum_qual:
          prev_j, prev_i, prev_cluster_index, prev_qual, best_prev_cum_qual = prev_node
      cum_qual = best_prev_cum_qual + qual
      prev_cache[int(j)] = (j, i, cluster_index, qual, cum_qual)
      cum_qual_jump = cum_qual - 1000
      if best_so_far[cur_index-1][4] < cum_qual_jump:
        while (cur_index < len(best_so_far)) and (best_so_far[cur_index][4] <= cum_qual_jump):
          del best_so_far[cur_index]
        best_so_far.add((j, i, cluster_index, qual, cum_qual_jump))
      if forward_min[i] == j and cur_index > 1:
        del best_so_far[:cur_index-1]
      cum_qual_cluster_jump = cum_qual - 50
      if cluster_last[3] < cum_qual_cluster_jump:
        clusters_best_so_far[cluster_index] = (j, i, qual, cum_qual_cluster_jump)
      backpointers[(j, i)] = (prev_j, prev_i, prev_cluster_index, prev_qual, best_prev_cum_qual)
  path = [best_so_far[-1]]
  while path[-1][:2] in backpointers:
    path.append(backpointers[path[-1][:2]])
  path.pop()
  path.reverse()
  path = np.array(path)
  y, x, cluster_indices, quals, cum_quals = path.T
  
  nondescription = ((quals == 0) | (quals > .3))
  similarity_ratio_x = float(len(set(x[nondescription]))) / len(audio_desc_features_scaled)
  similarity_ratio_y = float(len(set(y[nondescription]))) / len(video_features_scaled)
  similarity_percent = 100 * max(similarity_ratio_x, similarity_ratio_y)
  
  nodes = []
  if cluster_indices[0] == cluster_indices[1]:
    nodes.append((x[0], y[0]))
  for i in range(len(x) - 1):
    if cluster_indices[i] != cluster_indices[i+1]:
      nodes.append((x[i] - .1, y[i] - .1))
      nodes.append((x[i+1] + .1, y[i+1] + .1))
  if cluster_indices[-2] == cluster_indices[-1]:
    nodes.append((x[-1], y[-1]))
  x, y = np.array(nodes).T / 210.
  
  if (x[1] - x[0]) > 2:
    slope_start = (y[1] - y[0]) / (x[1] - x[0])
    x[0] = 0
    y[0] = y[1] - (x[1] * slope_start)
    if y[0] < 0:
      x[0] = x[1] - (y[1] / slope_start)
      y[0] = 0
  if (x[-1] - x[-2]) > 2:
    slope_end = (y[-1] - y[-2]) / (x[-1] - x[-2])
    x[-1] = ((len(audio_desc_energy) - 1) / 210.)
    y[-1] = y[-2] + ((x[-1] - x[-2]) * slope_end)
    if y[-1] > ((len(video_energy) - 1) / 210.):
      y[-1] = ((len(video_energy) - 1) / 210.)
      x[-1] = x[-2] + ((y[-1] - y[-2]) / slope_end)
  
  path[:,:2] /= 210.
  return x, y, similarity_percent, path, median_slope

# combines videos with matching audio files (e.g. audio descriptions)
# this is the main function of this script, it calls the other functions in order
def combine(video, audio, stretch_audio=False, yes=False, prepend="ad_", no_pitch_correction=False,
            output_dir=default_output_dir, alignment_dir=default_alignment_dir):
  video_files, has_audio_extensions = get_sorted_filenames(video, VIDEO_EXTENSIONS, AUDIO_EXTENSIONS)
  
  if yes == False and sum(has_audio_extensions) > 0:
    print("")
    print("One or more audio files found in video input. Was this intentional?")
    print("If not, press ctrl+c to kill this script.")
    input("If this was intended, press Enter to continue...")
    print("")
  audio_desc_files, _ = get_sorted_filenames(audio, AUDIO_EXTENSIONS)
  if len(video_files) != len(audio_desc_files):
    error_msg = ["Number of valid files in input paths are not the same.",
                 f"The video path has {len(video_files)} files",
                 f"The audio path has {len(audio_desc_files)} files"]
    raise RuntimeError("\n".join(error_msg))
  
  print("")
  ensure_folders_exist([output_dir])
  if PLOT_ALIGNMENT_TO_FILE:
    ensure_folders_exist([alignment_dir])
  
  print("")
  for (video_file, audio_desc_file) in zip(video_files, audio_desc_files):
    print(os.path.split(video_file)[1])
    print(os.path.split(audio_desc_file)[1])
    print("")
  if yes == False:
    print("Are the above input file pairings correct?")
    print("If not, press ctrl+c to kill this script.")
    input("If they are correct, press Enter to continue...")
    print("")
  
  # if ffmpeg isn't installed, install it
  if not is_ffmpeg_installed():
    print("Downloading and installing ffmpeg (media editor, 50 MB download)...")
    get_ffmpeg()
    if not is_ffmpeg_installed():
      RuntimeError("Failed to install ffmpeg.")
    print("Successfully installed ffmpeg.")
  
  print("Processing files:")
  
  for (video_file, audio_desc_file, has_audio_extension) in zip(video_files, audio_desc_files,
                                                           has_audio_extensions):
    # Output filename (and extension) is the same as input, except the prepend and directory
    output_filename = prepend + os.path.split(video_file)[1]
    output_filename = os.path.join(output_dir, output_filename)
    print(f" {output_filename}")
    
    if (not stretch_audio) & has_audio_extension:
      raise RuntimeError("Argument --stretch_audio is required when both inputs are audio files.")
    
    if os.path.exists(output_filename) and os.path.getsize(output_filename) > 1e5:
      print("   output file already exists, skipping...")
      continue
    
    # print warning if output file's full path is longer than Windows MAX_PATH (260)
    full_output_filename = os.path.abspath(output_filename)
    if IS_RUNNING_WINDOWS and len(full_output_filename) >= 260:
      print("  WARNING: very long output path, ffmpeg may fail...")
    
    num_channels = 2 if stretch_audio else 1
    print("  reading video file...\r", end='')
    video_arr = parse_audio_from_file(video_file, num_channels)
    
    print("  computing video features... \r", end='')
    video_energy = get_energy(video_arr)
    video_zero_crossings = get_zero_crossings(video_arr)
    video_freq_bands = get_freq_bands(video_arr)
    video_features = [video_energy, video_zero_crossings] + video_freq_bands
    
    if not stretch_audio:
      del video_arr
    
    print("  reading audio file...       \r", end='')
    audio_desc_arr = parse_audio_from_file(audio_desc_file, num_channels)
    
    print("  computing audio features...\r", end='')
    audio_desc_energy = get_energy(audio_desc_arr)
    audio_desc_zero_crossings = get_zero_crossings(audio_desc_arr)
    audio_desc_freq_bands = get_freq_bands(audio_desc_arr)
    audio_desc_features = [audio_desc_energy, audio_desc_zero_crossings] + audio_desc_freq_bands
    
    if not stretch_audio:
      del audio_desc_arr
    
    outputs = align(video_features, audio_desc_features, video_energy, audio_desc_energy)
    audio_desc_times, video_times, similarity_percent, path, median_slope = outputs
    
    del video_energy, video_zero_crossings, video_freq_bands, video_features
    del audio_desc_energy, audio_desc_zero_crossings, audio_desc_freq_bands, audio_desc_features
    
    if similarity_percent < 20:
      print(f"  WARNING: similarity {similarity_percent:.1f}%, likely mismatched files")
    if similarity_percent > 90:
      print(f"  WARNING: similarity {similarity_percent:.1f}%, likely undescribed media")
    
    if stretch_audio:
      # lower memory usage version of np.std for large arrays
      def low_ram_std(arr):
        avg = np.mean(arr, dtype=np.float64)
        return np.sqrt(np.einsum('ij,ij->i', arr, arr, dtype=np.float64)/np.prod(arr.shape) - (avg**2))
      
      # rescale RMS intensity of audio to match video
      audio_desc_arr *= (low_ram_std(video_arr) / low_ram_std(audio_desc_arr))[:, None]
      
      replace_aligned_segments(video_arr, audio_desc_arr, audio_desc_times, video_times, no_pitch_correction)
      del audio_desc_arr
      
      # prevent peaking by rescaling to within +/- 32,766
      video_arr *= (2**15 - 2.) / np.max(np.abs(video_arr))
      
      print("  processing output file...                   \r", end='')
      write_replaced_media_to_disk(output_filename, video_arr, None if has_audio_extension else video_file)
      del video_arr
    else:
      video_offset = video_times[0] - audio_desc_times[0]
      # to make ffmpeg cut at the last keyframe before the audio starts, use a timestamp after it
      after_start_key_frame = get_closest_key_frame_time(video_file, video_offset)
      print("  processing output file...                   \r", end='')
      setts_cmd = encode_fit_as_ffmpeg_expr(audio_desc_times, video_times, video_offset)
      write_replaced_media_to_disk(output_filename, None, video_file, audio_desc_file,
                                   setts_cmd, video_offset, after_start_key_frame)
    
    if PLOT_ALIGNMENT_TO_FILE:
      plot_filename_no_ext = os.path.join(alignment_dir, os.path.splitext(os.path.split(video_file)[1])[0])
      plot_alignment(plot_filename_no_ext, path, audio_desc_times, video_times, similarity_percent,
                     median_slope, stretch_audio, no_pitch_correction)
  print("All files processed.       ")

if wx is not None:
  def write_config_file(config_path, settings):
    config = configparser.ConfigParser()
    config.add_section('alignment')
    config['alignment'] = {}
    for key, value in settings.items():
      config['alignment'][key] = str(value)
    with open(config_path, 'w') as f:
      config.write(f)

  def read_config_file(config_path: Path):
    config = configparser.ConfigParser()
    config.read(config_path)
    settings = {'stretch_audio':       config.getboolean('alignment', 'stretch_audio', fallback=False),
                'prepend':             config.get('alignment', 'prepend', fallback='ad_'),
                'no_pitch_correction': config.getboolean('alignment', 'no_pitch_correction', fallback=False),
                'output_dir':          config.get('alignment', 'output_dir', fallback=default_output_dir),
                'alignment_dir':       config.get('alignment', 'alignment_dir', fallback=default_alignment_dir)}
    if not config.has_section('alignment'):
      write_config_file(config_path, settings)
    return settings
  
  def set_tooltip(element, tip):
    element.SetToolTip(tip)
    # prevent tooltip from disappearing for 30 seconds
    tooltip_object = element.GetToolTip()
    if not tooltip_object is None:
      tooltip_object.SetAutoPop(30000)
  
  class DialogSettings(wx.Dialog):
    def __init__(self, parent, config_path, is_dark):
      wx.Dialog.__init__(self, parent, title="Settings - describealign", size=wx.Size(450,330), 
                         style=wx.DEFAULT_DIALOG_STYLE|wx.TAB_TRAVERSAL)
      # setting the GUI dialog's font causes all contained elements to inherit that font by default
      self.SetFont(wx.Font(*gui_font))
      self.SetBackgroundColour(gui_background_color_dark if is_dark else gui_background_color_light)
      
      self.text_header = wx.StaticText(self, label="Check tooltips (i.e. mouse-over text) for descriptions:")
      
      self.static_box_sizer_output = wx.StaticBoxSizer(wx.VERTICAL, self, "output_dir")
      self.dir_picker_output = wx.DirPickerCtrl(self, message="Select a folder", name="output_dir")
      set_tooltip(self.dir_picker_output, "Directory combined output media is saved to. " + \
                                          "Default is \"videos_with_ad\"")
      
      self.static_box_sizer_alignment = wx.StaticBoxSizer(wx.VERTICAL, self, "alignment_dir")
      self.dir_picker_alignment = wx.DirPickerCtrl(self, message="Select a folder", name="alignment_dir")
      set_tooltip(self.dir_picker_alignment, "Directory alignment data and plots are saved to. " + \
                                             "Default is \"alignment_plots\"")
      
      self.text_prepend = wx.StaticText(self, label="prepend:")
      self.text_ctrl_prepend = wx.TextCtrl(self, name="prepend")
      set_tooltip(self.text_ctrl_prepend, "Output file name prepend text. Default is \"ad_\"")
      
      self.checkbox_stretch_audio = wx.CheckBox(self, label="stretch_audio", name="stretch_audio")
      set_tooltip(self.checkbox_stretch_audio, "Stretches the input audio to fit the input video. " + \
                                               "Default is to stretch the video to fit the audio. " + \
                                               "Keeps original video audio as secondary tracks. Slower " + \
                                               "and uses more RAM when enabled, long videos may cause " + \
                                               "paging or Out of Memory errors on low-RAM systems.")
      self.checkbox_stretch_audio.Bind(wx.EVT_CHECKBOX, self.update_stretch_audio_subsettings)
      
      self.checkbox_no_pitch_correction = wx.CheckBox(self, label="no_pitch_correction",
                                                      name="no_pitch_correction")
      set_tooltip(self.checkbox_no_pitch_correction, "Skips pitch correction step when stretching audio. " + \
                                                     "Requires --stretch_audio to be set, otherwise " + \
                                                     "does nothing.")
      
      self.button_save = wx.Button(self, label="Save")
      self.button_save.Bind(wx.EVT_BUTTON, self.save_settings)
      self.button_cancel = wx.Button(self, label="Cancel")
      self.button_cancel.Bind(wx.EVT_BUTTON, lambda event: self.EndModal(0))
      
      sizer_dialog = wx.BoxSizer(wx.VERTICAL)
      sizer_output_dir = wx.BoxSizer(wx.HORIZONTAL)
      sizer_alignment_dir = wx.BoxSizer(wx.HORIZONTAL)
      sizer_prepend = wx.BoxSizer(wx.HORIZONTAL)
      sizer_stretch_audio_no_pitch_correction = wx.BoxSizer(wx.VERTICAL)
      sizer_save_cancel = wx.BoxSizer(wx.HORIZONTAL)
      
      # Configure layout with nested Box Sizers:
      #
      # Frame
      #   sizer_dialog
      #     text_header
      #     sizer_output_dir
      #       static_box_sizer_output
      #         dir_picker_output
      #     sizer_alignment_dir
      #       static_box_sizer_alignment
      #         dir_picker_alignment
      #     sizer_prepend
      #       text_prepend
      #       text_ctrl_prepend
      #     sizer_stretch_audio_no_pitch_correction
      #       checkbox_stretch_audio
      #       checkbox_no_pitch_correction
      #     sizer_save_cancel
      #       button_save
      #       button_cancel
      #
      self.SetSizer(sizer_dialog)
      sizer_dialog.Add(self.text_header, 0, wx.ALL, 5)
      sizer_dialog.Add(sizer_output_dir, 1, wx.LEFT|wx.RIGHT|wx.EXPAND, 2)
      sizer_dialog.Add(sizer_alignment_dir, 1, wx.LEFT|wx.RIGHT|wx.EXPAND, 2)
      sizer_dialog.Add(sizer_prepend, 1, wx.LEFT|wx.EXPAND, 5)
      sizer_dialog.Add(sizer_stretch_audio_no_pitch_correction, 1, wx.LEFT|wx.EXPAND, 5)
      sizer_dialog.Add(sizer_save_cancel, 2, wx.BOTTOM|wx.EXPAND, 5)
      sizer_prepend.Add(self.text_prepend, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
      sizer_prepend.Add(self.text_ctrl_prepend, 0, wx.ALIGN_CENTER_VERTICAL, 5)
      sizer_output_dir.Add(self.static_box_sizer_output, 1, wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
      self.static_box_sizer_output.Add(self.dir_picker_output, 1, wx.EXPAND)
      sizer_alignment_dir.Add(self.static_box_sizer_alignment, 1, wx.LEFT|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 5)
      self.static_box_sizer_alignment.Add(self.dir_picker_alignment, 1, wx.EXPAND)
      sizer_stretch_audio_no_pitch_correction.Add(self.checkbox_stretch_audio, 0, wx.ALL, 5)
      sizer_stretch_audio_no_pitch_correction.Add(self.checkbox_no_pitch_correction, 0, wx.ALL, 5)
      sizer_save_cancel.Add((0, 0), 3, wx.EXPAND, 5)  # spacer
      sizer_save_cancel.Add(self.button_save, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
      sizer_save_cancel.Add((0, 0), 2, wx.EXPAND, 5)  # spacer
      sizer_save_cancel.Add(self.button_cancel, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
      sizer_save_cancel.Add((0, 0), 3, wx.EXPAND, 5)  # spacer
      
      # centers GUI on the screen
      self.Centre(wx.BOTH)
      
      # cache dictionaries mapping setting names to widget setter and getter functions
      self.setting_getters = {}
      self.setting_setters = {}
      for child in self.GetChildren():
        child_class_name = child.GetClassName()
        child_name = child.GetName()
        if child_class_name == "wxDirPickerCtrl":
          self.setting_getters[child_name] = child.GetPath
          self.setting_setters[child_name] = child.SetPath
        if child_class_name in ["wxCheckBox"]:
          self.setting_getters[child_name] = child.GetValue
          self.setting_setters[child_name] = child.SetValue
        if child_class_name in ["wxTextCtrl"]:
          self.setting_getters[child_name] = child.GetValue
          self.setting_setters[child_name] = lambda value, child=child: child.SetValue(str(value))
      self.setting_names = self.setting_getters.keys()
      
      # initialize setting widgets to saved config values
      self.config_path = config_path
      config_file_settings = read_config_file(self.config_path)
      for setting_name in self.setting_names:
        self.setting_setters[setting_name](config_file_settings[setting_name])

      # initialize stretch_audio subsettings to be disabled/enabled
      self.update_stretch_audio_subsettings()
      
      set_background_color(self, is_dark)
    
    def update_stretch_audio_subsettings(self, event=None):
      subsettings = [self.checkbox_no_pitch_correction]
      if self.checkbox_stretch_audio.IsChecked():
        for subsetting in subsettings:
          subsetting.Enable()
      else:
        for subsetting in subsettings:
          subsetting.Disable()
    
    def save_settings(self, event):
      settings = {}
      for setting_name in self.setting_names:
        settings[setting_name] = self.setting_getters[setting_name]()
      write_config_file(self.config_path, settings)
      self.EndModal(0)

  class QueueWriter(io.TextIOWrapper):
    def __init__(self, queue) -> None:
      super().__init__(buffer=io.BytesIO())
      self._queue = queue
      
    def write(self, s: str) -> int:
      self._queue.put(s)
      return len(s)

  def combine_print_exceptions(print_queue, *args, **kwargs):
    writer = QueueWriter(print_queue)
    with redirect_stdout(writer), redirect_stderr(writer):
      try:
        combine(*args, **kwargs)
      except Exception:
        print("  ERROR: exception raised")
        traceback.print_exc()

  class FrameCombine(wx.Frame):
    def __init__(self, parent, config_path, video_files, audio_files, is_dark):
      wx.Frame.__init__(self, parent, title="Combining - describealign", size=wx.Size(800,600))
      # setting the GUI frame's font causes all contained elements to inherit that font by default
      self.SetFont(wx.Font(*gui_font))
      self.SetBackgroundColour(gui_background_color_dark if is_dark else gui_background_color_light)
      # wrap all widgets within a panel to enable tab traversal (i.e. pressing tab to swap GUI focus)
      self.panel0 = wx.Panel(self, style=wx.TAB_TRAVERSAL)
      
      self.text_ctrl_output = wx.TextCtrl(self.panel0, style=wx.TE_MULTILINE|wx.TE_READONLY|wx.TE_RICH)
      
      self.button_close = wx.Button(self.panel0, label="Close")
      self.button_close.Bind(wx.EVT_BUTTON, self.attempt_close)
      # also capture other close events such as alt+f4 or clicking the X in the top corner of the frame
      self.Bind(wx.EVT_CLOSE, self.attempt_close)
      
      self.update_timer = wx.Timer(self)
      self.Bind(wx.EVT_TIMER, self.update_gui, self.update_timer)
      
      sizer_panel_outer = wx.BoxSizer(wx.VERTICAL)
      sizer_panel_inner = wx.BoxSizer(wx.VERTICAL)
      sizer_close_button = wx.BoxSizer(wx.HORIZONTAL)
      
      # Configure layout with nested Box Sizers:
      #
      # Frame
      #   sizer_panel_outer
      #     panel0
      #       sizer_panel_inner
      #         text_ctrl_output
      #         sizer_close_button
      #           button_close
      #
      self.SetSizer(sizer_panel_outer)
      sizer_panel_outer.Add(self.panel0, 1, wx.EXPAND|wx.ALL, 5)
      self.panel0.SetSizer(sizer_panel_inner)
      sizer_panel_inner.Add(self.text_ctrl_output, 1, wx.ALL|wx.EXPAND, 5)
      sizer_panel_inner.Add(sizer_close_button, 0, wx.EXPAND, 5)
      sizer_close_button.Add((0, 0), 1, wx.EXPAND, 5)  # spacer
      sizer_close_button.Add(self.button_close, 0, wx.ALL, 5)
      sizer_close_button.Add((0, 0), 1, wx.EXPAND, 5)  # spacer
      
      # centers GUI on the screen
      self.Centre(wx.BOTH)
      
      set_background_color(self, is_dark)
      
      self.config_path = config_path
      self.overwrite_last_line = False
      self.display_line('Combining media files:')
      self.text_ctrl_output.SetInsertionPoint(0)
      
      # launch combiner using settings from config file, redirecting its output to a queue
      self.print_queue = multiprocessing.Queue()
      settings = read_config_file(self.config_path)
      settings.update({'yes':True})
      self.combine_process = multiprocessing.Process(target=combine_print_exceptions,
                                                     args=(self.print_queue, video_files, audio_files),
                                                     kwargs=settings, daemon=True)
      self.combine_process.start()
      self.update_gui()
    
    def attempt_close(self, event):
      if self.combine_process.is_alive():
        dialog = wx.MessageDialog(self, "Warning: combiner is still running, stop it and close anyway?",
                                  "Warning", wx.YES_NO|wx.ICON_WARNING)
        response = dialog.ShowModal()
        if (response == wx.ID_YES):
          self.combine_process.terminate()
          self.Destroy()
        elif (response == wx.ID_NO):
          # If the EVT_CLOSE came from the OS, let the OS know it didn't succeed
          if event.GetEventType() == wx.EVT_CLOSE.evtType[0]:
            event.Veto(True)
      else:
        self.Destroy()
    
    def set_last_line_color(self, color, line_start):
      num_lines = self.text_ctrl_output.GetNumberOfLines()
      end = self.text_ctrl_output.GetLastPosition()
      self.text_ctrl_output.SetStyle(line_start, end, wx.TextAttr("black", color))
    
    def display_line(self, line):
      if self.overwrite_last_line:
        # skip the empty line following lines ending in "\r"
        if line == "":
          return
        num_lines = self.text_ctrl_output.GetNumberOfLines()
        start = self.text_ctrl_output.XYToPosition(0,num_lines-2)
        end = self.text_ctrl_output.GetLastPosition()
        self.text_ctrl_output.Remove(start, end)
        self.overwrite_last_line = False
      if line[-1:] == "\r":
        self.overwrite_last_line = True
        line = line[:-1].rstrip(' ') + "\r"
      line_start = self.text_ctrl_output.GetLastPosition()
      self.text_ctrl_output.AppendText(line)
      # highlight warnings by changing their background color to light orange
      if line[:10] == "  WARNING:":
        self.set_last_line_color(wx.Colour(255, 188, 64), line_start)
      # highlight errors by changing their background color to red
      if line[:8] == "  ERROR:":
        self.set_last_line_color(wx.Colour(255, 128, 128), line_start)
    
    def update_gui(self, event=None):
      lines = []
      while not self.print_queue.empty():
        lines.append(self.print_queue.get())
      if len(lines) > 0:
        cursor_position = self.text_ctrl_output.GetInsertionPoint()
        self.text_ctrl_output.Freeze()
        for line in lines:
          self.display_line(line)
        self.text_ctrl_output.SetInsertionPoint(cursor_position)
        self.text_ctrl_output.Thaw()
      self.update_timer.StartOnce(gui_update_interval_ms)

  def migrate_config(old_path: Optional[Path], new_path: Path) -> None:
    """
    Migrate configuration from old location.
    
    Only runs if the old_path exists but new_path does not
    """
    if new_path.exists() or not old_path or not old_path.exists():
      return
    
    old_data = old_path.read_text(encoding='utf-8')
    new_path.write_text(old_data, encoding='utf-8')
    print(f"Configuration migrated to {new_path}")
    try:
      old_path.unlink()
    except OSError as exc:
      print("Failed to remove old config:", *traceback.format_exception_only(exc))
    else:
      print("Successfully removed old config file.")

  class ListCtrlDropTarget(wx.FileDropTarget):
    def __init__(self, list_ctrl, parent_frame):
      super().__init__()
      self.list_ctrl = list_ctrl
      self.parent_frame = parent_frame
    
    def expand_folders(self, files):
      expanded_files = []
      for file in files:
        if os.path.isdir(file):
          for dir, subdirs, dir_files in os.walk(file):
            for dir_file in dir_files:
              expanded_files.append(os.path.join(dir, dir_file))
        else:
          expanded_files.append(file)
      return expanded_files
    
    def OnDropFiles(self, x, y, files):
      files = self.expand_folders(files)
      valid_file_types = self.parent_frame.list_ctrl_file_types_drop[self.list_ctrl]
      files = [file for file in files if os.path.splitext(file)[-1][1:] in valid_file_types]
      self.parent_frame.populate_list_ctrl(self.list_ctrl, natsort.os_sorted(files))
      return True

  def get_children(window):
    children = list(window.GetChildren())
    subchildren = [subchild for child in children for subchild in get_children(child)]
    return children + subchildren

  def set_background_color(window, is_dark):
    children = get_children(window)
    for window in children + [window]:
      if is_dark:
        if isinstance(window, (wx.ListCtrl, wx.Button, wx.TextCtrl)):
          window.SetBackgroundColour("Black")
        else:
          window.SetBackgroundColour(gui_background_color_dark)
      window.SetForegroundColour("White" if is_dark else "Black")

  class FrameMain(wx.Frame):
    def __init__(self, parent):
      wx.Frame.__init__(self, parent, title="describealign", size=wx.Size(800, 500))
      # setting the GUI frame's font causes all contained elements to inherit that font by default
      self.SetFont(wx.Font(*gui_font))
      appearance = wx.SystemSettings.GetAppearance()
      self.is_dark = appearance.IsDark() or appearance.IsUsingDarkBackground()
      self.SetBackgroundColour(gui_background_color_dark if self.is_dark else gui_background_color_light)
      
      # wrap all widgets within a panel to enable tab traversal (i.e. pressing tab to swap GUI focus)
      self.panel0 = wx.Panel(self, style=wx.TAB_TRAVERSAL)
      
      self.text_header = wx.StaticText(self.panel0, label="Select media files to combine:")
      self.text_header.SetFont(self.text_header.GetFont().Scale(1.7))
      
      # Video Input selection and display row of GUI
      self.static_box_sizer_video = wx.StaticBoxSizer(wx.HORIZONTAL, self.panel0, "Video Input")
      self.list_ctrl_video = self.init_list_ctrl(self.static_box_sizer_video.GetStaticBox(),
                                                 "Drag and Drop Videos Here or Press Browse Video")
      set_tooltip(self.list_ctrl_video, "Video filenames are listed here in the sorted order they will " + \
                                        "be used as input. Drag and Drop or press Browse to overwrite.")
      self.button_browse_video = wx.Button(self.static_box_sizer_video.GetStaticBox(), label="Browse Video")
      set_tooltip(self.button_browse_video, "Select one or more video files as input.")
      self.button_browse_video.Bind(wx.EVT_BUTTON, lambda event: self.browse_files(self.list_ctrl_video))
      
      # Audio Input selection and display row of GUI
      self.static_box_sizer_audio = wx.StaticBoxSizer(wx.HORIZONTAL, self.panel0, "Audio Input")
      self.list_ctrl_audio = self.init_list_ctrl(self.static_box_sizer_audio.GetStaticBox(),
                                                 "Drag and Drop Audio Here or Press Browse Audio")
      set_tooltip(self.list_ctrl_audio, "Audio filenames are listed here in the sorted order they will " + \
                                        "be used as input. Drag and Drop or press Browse to overwrite.")
      self.button_browse_audio = wx.Button(self.static_box_sizer_audio.GetStaticBox(), label="Browse Audio")
      set_tooltip(self.button_browse_audio, "Select one or more audio files as input.")
      self.button_browse_audio.Bind(wx.EVT_BUTTON, lambda event: self.browse_files(self.list_ctrl_audio))
      
      self.button_combine = wx.Button(self.panel0, label="Combine")
      set_tooltip(self.button_combine, "Combine selected video and audio files.")
      self.button_combine.Bind(wx.EVT_BUTTON, self.open_combine)
      self.button_settings = wx.Button(self.panel0, label="Settings")
      set_tooltip(self.button_settings, "Edit settings for the GUI and algorithm.")
      self.button_settings.Bind(wx.EVT_BUTTON, self.open_settings)
      
      sizer_panel_outer = wx.BoxSizer(wx.VERTICAL)
      sizer_panel_inner = wx.BoxSizer(wx.VERTICAL)
      sizer_header = wx.BoxSizer(wx.HORIZONTAL)
      sizer_video = wx.BoxSizer(wx.HORIZONTAL)
      sizer_audio = wx.BoxSizer(wx.HORIZONTAL)
      sizer_combine_settings = wx.BoxSizer(wx.HORIZONTAL)
      
      # Configure layout with nested Box Sizers:
      #
      # Frame
      #   sizer_panel_outer
      #     panel0
      #       sizer_panel_inner
      #         sizer_header
      #           text_header
      #         sizer_video
      #           list_ctrl_video
      #           button_browse_video
      #         sizer_audio
      #           list_ctrl_audio
      #           button_browse_audio
      #         sizer_combine_settings
      #           button_combine
      #           button_settings
      #
      self.SetSizer(sizer_panel_outer)
      sizer_panel_outer.Add(self.panel0, 1, wx.EXPAND|wx.ALL, 5)
      self.panel0.SetSizer(sizer_panel_inner)
      sizer_panel_inner.Add(sizer_header, 3, wx.EXPAND, 5)
      sizer_panel_inner.Add(sizer_video, 9, wx.EXPAND, 5)
      sizer_panel_inner.Add(sizer_audio, 9, wx.TOP|wx.EXPAND, 3)
      sizer_panel_inner.Add(sizer_combine_settings, 3, wx.EXPAND, 5)
      sizer_header.Add(self.text_header, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
      sizer_video.Add(self.static_box_sizer_video, 1, wx.LEFT|wx.RIGHT|wx.EXPAND, 3)
      self.static_box_sizer_video.Add(self.list_ctrl_video, 1, wx.BOTTOM|wx.EXPAND, 2)
      self.static_box_sizer_video.Add(self.button_browse_video, 0,
                                      wx.LEFT|wx.BOTTOM|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 10)
      sizer_audio.Add(self.static_box_sizer_audio, 1, wx.LEFT|wx.RIGHT|wx.EXPAND, 3)
      self.static_box_sizer_audio.Add(self.list_ctrl_audio, 1, wx.BOTTOM|wx.EXPAND, 2)
      self.static_box_sizer_audio.Add(self.button_browse_audio, 0,
                                      wx.LEFT|wx.BOTTOM|wx.RIGHT|wx.ALIGN_CENTER_VERTICAL, 10)
      sizer_combine_settings.Add((0, 0), 7, wx.EXPAND, 5)  # spacer
      sizer_combine_settings.Add(self.button_combine, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
      sizer_combine_settings.Add((0, 0), 2, wx.EXPAND, 5)  # spacer
      sizer_combine_settings.Add(self.button_settings, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
      sizer_combine_settings.Add((0, 0), 7, wx.EXPAND, 5)  # spacer
      
      # centers GUI on the screen
      self.Centre(wx.BOTH)
      
      all_video_file_types = [('All Video File Types', '*.' + ';*.'.join(VIDEO_EXTENSIONS)),]
      all_audio_file_types = [('All Audio File Types', '*.' + ';*.'.join(AUDIO_EXTENSIONS)),]
      all_video_and_audio_file_types = [('All Video and Audio File Types',
                                         '*.' + ';*.'.join(VIDEO_EXTENSIONS | AUDIO_EXTENSIONS)),]
      self.video_file_types = [(ext, f"*.{ext}") for ext in VIDEO_EXTENSIONS]
      self.audio_file_types = [(ext, f"*.{ext}") for ext in AUDIO_EXTENSIONS]
      self.video_and_audio_file_types = self.video_file_types + self.audio_file_types
      self.video_file_types = all_video_file_types + self.video_file_types
      self.audio_file_types = all_audio_file_types + self.audio_file_types
      self.video_and_audio_file_types = all_video_file_types + all_video_and_audio_file_types + \
                                        self.video_and_audio_file_types
      self.video_file_types = '|'.join([f'{type[0]} ({type[1]})|{type[1]}' for type in self.video_file_types])
      self.audio_file_types = '|'.join([f'{type[0]} ({type[1]})|{type[1]}' for type in self.audio_file_types])
      self.video_and_audio_file_types = '|'.join([f'{type[0]} ({type[1]})|{type[1]}' for type \
                                                  in self.video_and_audio_file_types])
      
      # track the allowed file types and selected files' full paths for each List Ctrl
      self.list_ctrl_file_types_browse = {self.list_ctrl_video: self.video_and_audio_file_types,
                                          self.list_ctrl_audio: self.audio_file_types}
      self.list_ctrl_file_types_drop = {self.list_ctrl_video: self.video_file_types,
                                        self.list_ctrl_audio: self.audio_file_types}
      self.list_ctrl_files_selected = {self.list_ctrl_video: [],
                                       self.list_ctrl_audio: []}
      
      self.config_path = self.get_config()
      
      set_background_color(self, self.is_dark)
    
    def init_list_ctrl(self, parent_panel, default_text):
      list_ctrl = wx.ListCtrl(parent_panel, style=wx.LC_NO_HEADER|wx.LC_REPORT|wx.BORDER_SUNKEN|wx.HSCROLL)
      list_ctrl.EnableSystemTheme(False)  # get rid of vertical grid lines on Windows
      list_ctrl.SetMinSize(wx.Size(-1,80))
      list_ctrl.SetDropTarget(ListCtrlDropTarget(list_ctrl, self))
      list_ctrl.InsertColumn(0, "")
      list_ctrl.InsertItem(0, default_text)
      list_ctrl.SetColumnWidth(0, wx.LIST_AUTOSIZE)
      list_ctrl.Bind(wx.EVT_CHAR, self.delete_from_list_ctrl)
      return list_ctrl
    
    def populate_list_ctrl(self, list_ctrl, files):
      self.list_ctrl_files_selected[list_ctrl] = files
      if len(files) == 0:
        files = ["No files with valid file types found"]
      list_ctrl.DeleteAllItems()
      list_ctrl.DeleteAllColumns()
      list_ctrl.InsertColumn(0, "")
      for i, file in enumerate(files):
        list_ctrl.InsertItem(i, os.path.basename(file))
      list_ctrl.SetColumnWidth(0, wx.LIST_AUTOSIZE)
    
    def browse_files(self, list_ctrl):
      dialog = wx.FileDialog(self, wildcard=self.list_ctrl_file_types_browse[list_ctrl], style=wx.FD_MULTIPLE)
      if dialog.ShowModal() == wx.ID_OK:
        files = dialog.GetPaths()
        self.populate_list_ctrl(list_ctrl, files)
    
    def delete_from_list_ctrl(self, event):
      if event.GetKeyCode() == wx.WXK_DELETE:
        list_ctrl = event.GetEventObject()
        item_index = list_ctrl.GetFirstSelected()
        if item_index == -1:
          item_index = list_ctrl.GetFocusedItem()
        items_to_delete = []
        while item_index != -1:
          items_to_delete.append(item_index)
          item_index = list_ctrl.GetNextSelected(item_index)
        for item_index in items_to_delete[::-1]:
          if len(self.list_ctrl_files_selected[list_ctrl]) != 0:
            list_ctrl.DeleteItem(item_index)
            del self.list_ctrl_files_selected[list_ctrl][item_index]
      else:
        event.Skip()
    
    def open_combine(self, event):
      video_files = self.list_ctrl_files_selected[self.list_ctrl_video]
      audio_files = self.list_ctrl_files_selected[self.list_ctrl_audio]
      if len(video_files) == 0:
        error_dialog = wx.MessageDialog(self, "Error: no video input selected.", "Error", wx.OK|wx.ICON_ERROR)
        error_dialog.ShowModal()
      elif len(audio_files) == 0:
        error_dialog = wx.MessageDialog(self, "Error: no audio input selected.", "Error", wx.OK|wx.ICON_ERROR)
        error_dialog.ShowModal()
      elif len(video_files) != len(audio_files):
        error_dialog = wx.MessageDialog(self, f"Error: different numbers of video ({len(video_files)}) " + \
                                              f"and audio ({len(audio_files)}) inputs.",
                                        "Error", wx.OK|wx.ICON_ERROR)
        error_dialog.ShowModal()
      else:
        frame_combine = FrameCombine(None, self.config_path, video_files, audio_files, self.is_dark)
        self.list_ctrl_video.SetFocus()
        frame_combine.Show()
    
    def open_settings(self, event):
      dialog_settings = DialogSettings(None, self.config_path, self.is_dark)
      dialog_settings.ShowModal()
      dialog_settings.Destroy()
    
    def get_config(self):
      config_path = platformdirs.user_config_path(appname='describealign', appauthor=False,
                                                  ensure_exists=True) / 'config.ini'
      old_paths = [
        # Place in chronological order (oldest -> newest)
        Path(__file__).resolve().parent / 'config.ini',
        platformdirs.user_config_path(appname='describealign', ensure_exists=True) / 'config.ini',
      ]
      # Get newest existent path
      old_config = next((file for file in reversed(old_paths) if file.exists()), None,)
      try:
        migrate_config(old_config, config_path)
      except OSError as exc:
        print(f"Error migrating old config:", *traceback.format_exception_only(exc))
        print(f"Old config left in place at {old_config}")
      return config_path

def get_version_hash(filename):
  try:
    with open(filename, 'rb') as f:
      data = f.read()
      sha_hash = hashlib.sha1(data).hexdigest()
    return sha_hash[:8]
  except:
    return "None"

# Entry point for command line interaction, for example:
# > describealign video.mp4 audio_desc.mp3
def command_line_interface():
  if len(sys.argv) < 2:
    if wx is not None:
      # No args, run gui
      print('No input arguments detected, starting GUI...')
      # the following line is necessary on MacOS X to fix the filectrlpicker
      # https://docs.wxpython.org/wx.FileDialog.html#wx-filedialog
      # https://github.com/wxWidgets/Phoenix/issues/2368
      if platform.system() == 'Darwin':
        wx.SystemOptions.SetOption('osx.openfiledialog.always-show-types', 1)
      app = wx.App()
      main_gui = FrameMain(None)
      main_gui.Show()
      app.MainLoop()
      sys.exit(0)
    else:
      print("Can't launch GUI and arguments missing.\nGUI dependencies missing.")
  
  parser = argparse.ArgumentParser(description="Replaces a video's sound with an audio description.",
                                   usage="describealign video_file.mp4 audio_file.mp3")
  parser.add_argument("video", help='A video file or directory containing video files.',
                      nargs='?', default=None)
  parser.add_argument("audio", help='An audio file or directory containing audio files.',
                      nargs='?', default=None)
  parser.add_argument('--stretch_audio', action='store_true',
                      help='Stretches the input audio to fit the input video. ' + \
                           'Default is to stretch the video to fit the audio. ' + \
                           'Keeps original video audio as secondary tracks. Slower ' + \
                           'and uses more RAM when enabled, long videos may cause ' + \
                           'paging or Out of Memory errors on low-RAM systems.')
  parser.add_argument('--yes', action='store_true',
                      help='Auto-skips user prompts asking to verify information.')
  parser.add_argument("--prepend", default="ad_", help='Output file name prepend text. Default is "ad_"')
  parser.add_argument('--no_pitch_correction', action='store_true',
                      help='Skips pitch correction step when stretching audio. ' + \
                           'Requires --stretch_audio to be set, otherwise does nothing.')
  parser.add_argument("--output_dir", default=default_output_dir,
                      help='Directory combined output media is saved to. Default is "videos_with_ad"')
  parser.add_argument("--alignment_dir", default=default_alignment_dir,
                      help='Directory alignment data and plots are saved to. Default is "alignment_plots"')
  parser.add_argument("--install-ffmpeg", action="store_true",
                      help="Install the required ffmpeg binaries and then exit. This is meant to be " + \
                           "run from a privileged installer process (e.g. OS X Installer)")
  parser.add_argument('--version', action='store_true',
                      help='Checks and prints the installed version of describealign.')
  args = parser.parse_args()
  
  if args.version:
    import importlib
    cur_dir = os.getcwd()
    if sys.path[0] == cur_dir:
      # ignore describealign.py in current directory
      del sys.path[0]
      installed_spec = importlib.util.find_spec('describealign')
      sys.path = [cur_dir] + sys.path
    else:
      installed_spec = importlib.util.find_spec('describealign')
    if installed_spec is None:
      print("describealign is not installed")
    else:
      installed_path = os.path.abspath(installed_spec.origin)
      this_script_path = os.path.abspath(__file__)
      if installed_path != this_script_path:
        print("WARNING: describealign is not being run from the installed version")
        print(f"  installed path: {installed_path}")
        print(f"    content hash: {get_version_hash(installed_path)}")
        print(f"  this file path: {this_script_path}")
        print(f"    content hash: {get_version_hash(this_script_path)}")
      print(f"installed version: {importlib.metadata.version('describealign')}")
  elif args.install_ffmpeg:
    # Make sure the file is world executable
    os.chmod(get_ffmpeg(), 0o755)
    os.chmod(get_ffprobe(), 0o755)
  elif args.video and args.audio:
    combine(args.video, args.audio, args.stretch_audio, args.yes, args.prepend, 
            args.no_pitch_correction, args.output_dir, args.alignment_dir)
  else:
    parser.print_usage()

# allows the script to be run on its own, rather than through the package, for example:
# python3 describealign.py video.mp4 audio_desc.mp3
if __name__ == "__main__":
  multiprocessing.freeze_support()
  command_line_interface()




