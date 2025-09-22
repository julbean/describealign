# combines videos with matching audio files (e.g. audio descriptions)
# input: video or folder of videos and an audio file or folder of audio files
# output: videos in a folder "videos_with_ad", with aligned segments of the audio replaced
# this script aligns the new audio to the video using the video's old audio
# first, the video's sound and the audio file are both converted to spectrograms
# second, the two spectrograms are roughly aligned by finding their longest common subsequence
# third, the rough alignment is denoised through L1-Minimization
# fourth, the spectrogram alignments determine where the new audio replaces the old

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

TIMESTEP_SIZE_SECONDS = .16
TIMESTEP_OVERLAP_RATIO = .5
AUDIO_SAMPLE_RATE = 44100
MEL_COEFFS_PER_TIMESTEP = 25
DITHER_PERIOD_STEPS = 60
MIN_CORR_FOR_TOKEN_MATCH = .6
GAP_START_COST = 1.0
GAP_EXTEND_COST = -.01
GAP_EXTEND_DIAG_BONUS = -.01
SKIP_MATCH_COST = .1
MAX_RATE_RATIO_DIFF_ALIGN = .1
PREF_CUT_AT_GAPS_FACTOR = 5
MIN_DURATION_TO_REPLACE_SECONDS = 2
MIN_START_END_SYNC_TIME_SECONDS = 2
MAX_START_END_SYNC_ERR_SECONDS = .2
MAX_RATE_RATIO_DIFF_BOOST = .003
MIN_DESC_DURATION = .5
MAX_GAP_IN_DESC_SEC = 1.5
JUST_NOTICEABLE_DIFF_IN_FREQ_RATIO = .005
CATCHUP_RATE = 5

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
import python_speech_features as psf
import scipy.signal
import scipy.optimize
import scipy.interpolate
import scipy.ndimage as nd
import scipy.sparse
import pytsmod
import configparser
import traceback
import multiprocessing
import platform

IS_RUNNING_WINDOWS = platform.system() == 'Windows'
if IS_RUNNING_WINDOWS:
  default_output_dir = 'videos_with_ad'
  default_alignment_dir = 'alignment_plots'
else:
  default_output_dir = os.path.expanduser('~') + '/videos_with_ad'
  default_alignment_dir = os.path.expanduser('~') + '/alignment_plots'

try:
  if IS_RUNNING_WINDOWS:
    import PySimpleGUIWx as sg
  else:
    import PySimpleGUIQt as sg
except ImportError:
  sg = None

def display(text, func=None):
  if func:
    func(text)
  print(text)

def throw_runtime_error(text, func=None):
  if func:
    func(text)
  raise RuntimeError(text)

def ensure_folders_exist(dirs, display_func=None):
  for dir in dirs:
    if not os.path.isdir(dir):
      display(f"Directory not found, creating it: {dir}", display_func)
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
  files = sorted(files)
  file_types = [0 if os.path.splitext(file)[1][1:] in extensions else 1 for file in files]
  return files, file_types

# read audio from file with ffmpeg and convert to numpy array
def parse_audio_from_file(media_file):
  media_stream, _ = (ffmpeg
    .input(media_file)
    .output('-', format='s16le', acodec='pcm_s16le', ac=2, ar=AUDIO_SAMPLE_RATE, loglevel='fatal')
    .run(capture_stdout=True, cmd=get_ffmpeg())
  )
  media_arr = np.frombuffer(media_stream, np.int16).astype(np.float32).reshape((-1,2)).T
  return media_arr

# tokenize audio by transforming with a mel-frequency cepstrum (MFC)
def tokenize_audio(media_arr, rate=1):
  step_size_samples = psf.sigproc.round_half_up(TIMESTEP_SIZE_SECONDS * rate * AUDIO_SAMPLE_RATE)
  window_size_seconds = TIMESTEP_SIZE_SECONDS / TIMESTEP_OVERLAP_RATIO
  window_size_samples = psf.sigproc.round_half_up(window_size_seconds * AUDIO_SAMPLE_RATE)
  fft_size_samples = 2**int(np.ceil(np.log2(window_size_samples)))
  get_mfcc = lambda arr: psf.mfcc(np.mean(arr, axis=0),
                                  samplerate=AUDIO_SAMPLE_RATE,
                                  winlen=window_size_seconds,
                                  winstep=TIMESTEP_SIZE_SECONDS * rate,
                                  numcep=MEL_COEFFS_PER_TIMESTEP,
                                  nfilt=MEL_COEFFS_PER_TIMESTEP * 2,
                                  nfft=fft_size_samples,
                                  winfunc=scipy.signal.windows.hann)
  num_timesteps = max(1, ((media_arr.shape[1] - window_size_samples - 1) // step_size_samples) + 2)
  media_spec = np.zeros((num_timesteps, MEL_COEFFS_PER_TIMESTEP))
  chunk_size = 1000
  for chunk_index in np.arange(0, num_timesteps, chunk_size):
    chunk_bounds_samples = ((chunk_index                 ) * step_size_samples,
                            (chunk_index + chunk_size - 1) * step_size_samples + window_size_samples)
    media_spec[chunk_index:chunk_index+chunk_size] = get_mfcc(media_arr[:,slice(*chunk_bounds_samples)])
  '''
  # alternate python library's MFC implementation
  import librosa
  media_spec = librosa.feature.mfcc(y=np.mean(media_arr, axis=0),
                                    sr=AUDIO_SAMPLE_RATE,
                                    n_mfcc=MEL_COEFFS_PER_TIMESTEP,
                                    lifter=22,
                                    n_fft=fft_size_samples,
                                    hop_length=step_size_samples,
                                    win_length=window_size_samples,
                                    window=scipy.signal.windows.hann).T
  num_timesteps = media_spec.shape[0]
  '''
  timings_samples = window_size_samples/2. + step_size_samples * np.arange(num_timesteps)
  timings_seconds = timings_samples / AUDIO_SAMPLE_RATE
  return media_spec, timings_seconds

# same as tokenize_audio, but dithering the MFC window timings
# this allows for finer alignment by ameliorating discretization error
def tokenize_audio_dither(media_arr, slow_timings):
  # choose a relative step size slightly less than 1 to ameliorate quantization error
  # maximize alignment accuracy by using least approximable number with desired period
  # this is the continued fraction [0;1,N-2,1,1,1,...], where the trailing ones give phi
  fast_rate = 1. / (1 + 1. / (DITHER_PERIOD_STEPS - 2 + (np.sqrt(5) + 1) / 2.))
  fast_spec, fast_timings = tokenize_audio(media_arr, fast_rate)
  
  # prevent drift in difficult to align segments (e.g. describer speaking or quiet/droning segments)
  # by approximately equalizing the number of tokens per unit time between dithered and undithered
  # the dithered audio will have ~(1 + 1 / DITHER_PERIOD_STEPS) times as many tokens, so
  # this can be accomplished by simply deleting a token every DITHER_PERIOD_STEPS tokens
  fast_spec = np.delete(fast_spec, slice(DITHER_PERIOD_STEPS // 2, None, DITHER_PERIOD_STEPS), axis=0)
  fast_timings = np.delete(fast_timings, slice(DITHER_PERIOD_STEPS // 2, None, DITHER_PERIOD_STEPS))
  return fast_spec, fast_timings

# normalize along both time and frequency axes to allow comparing tokens by correlation
def normalize_spec(media_spec_raw, axes=(0,1)):
  media_spec = media_spec_raw.copy()
  for axis in axes:
    norm_func = np.std if axis == 0 else np.linalg.norm
    media_spec = media_spec - np.mean(media_spec, axis=axis, keepdims=True)
    media_spec = media_spec/(norm_func(media_spec,axis=axis,keepdims=True)+1e-10)
  return media_spec

# vectorized implementation of the Wagner–Fischer (Longest Common Subsequence) algorithm
# modified to include affine gap penalties and skip+match options (i.e. knight's moves)
# gaps are necessary when parts are cut out of the audio description (e.g. cut credits)
# or when the audio description includes a commercial break or an extra scene
# the skip+match option allows for micro-adjustments without eating the full gap penalty
# skip+match is primarily useful in maintaining alignment when the rates differ slightly
def rough_align(video_spec, audio_desc_spec, video_timings, audio_desc_timings):
  pred_map = {0:lambda node: (0, node[1]-1, node[2]-1),
              1:lambda node: (0, node[1]-2, node[2]-1),
              2:lambda node: (0, node[1]-1, node[2]-2),
              3:lambda node: (1, node[1]-1, node[2]-1),
              4:lambda node: (0, node[1]  , node[2]  ),
              5:lambda node: (1, node[1]-1, node[2]  ),
              6:lambda node: (1, node[1]-1, node[2]-1),
              7:lambda node: (1, node[1]  , node[2]-1)}
  pred_matrix = np.zeros((2, audio_desc_spec.shape[0], video_spec.shape[0]), dtype=np.uint8)
  pred_matrix[0,1:,:2] = 0
  pred_matrix[1,1:,:2] = 4
  pred_matrix[:,0,:2] = [0,5]
  path_corrs_match = np.zeros((3, video_spec.shape[0]))
  path_corrs_gap = np.zeros((3, video_spec.shape[0]))
  corrs = np.zeros((3, video_spec.shape[0]))
  corrs[:,:] = np.roll(np.dot(video_spec, audio_desc_spec[0]), 1)[None,:]
  for i in range(audio_desc_spec.shape[0]):
    i_mod = i % 3
    match_pred_corrs = np.hstack([path_corrs_match[i_mod-1][1:-1][:,None],
                                  path_corrs_match[i_mod-2][1:-1][:,None] - SKIP_MATCH_COST,
                                  path_corrs_match[i_mod-1][0:-2][:,None] - SKIP_MATCH_COST,
                                  path_corrs_gap[  i_mod-1][1:-1][:,None]])
    pred_matrix[0][i][2:] = np.argmax(match_pred_corrs, axis=1)
    path_corrs_match[i_mod][2:] = np.take_along_axis(match_pred_corrs, pred_matrix[0][i][2:,None], axis=1).T
    corrs = np.roll(corrs, -1, axis=1)
    corrs[(i_mod+1)%3,:] = np.roll(np.dot(video_spec, audio_desc_spec[min(audio_desc_spec.shape[0]-1,i+1)]), 1)
    fisher_infos = (2 * corrs[i_mod] - corrs[i_mod-1] - corrs[(i_mod+1)%3]) / min(.2, TIMESTEP_SIZE_SECONDS)
    fisher_infos[fisher_infos < 0] = 0
    fisher_infos[fisher_infos > 10] = 10
    row_corrs = np.maximum(0, corrs[i_mod][2:] - MIN_CORR_FOR_TOKEN_MATCH)
    path_corrs_match[i_mod][2:] += row_corrs * (fisher_infos[2:] / 5)
    gap_pred_corrs = np.hstack([path_corrs_match[i_mod][2:  ][:,None] - GAP_START_COST,
                                path_corrs_gap[i_mod-1][2:  ][:,None],
                                path_corrs_gap[i_mod-1][1:-1][:,None] - GAP_EXTEND_DIAG_BONUS - \
                                                                        GAP_EXTEND_COST])
    pred_matrix[1][i][2:] = np.argmax(gap_pred_corrs, axis=1)
    path_corrs_gap_no_col_skip = np.take_along_axis(gap_pred_corrs, pred_matrix[1][i][2:,None], axis=1).flat
    pred_matrix[1][i][2:] += 4
    path_corrs_gap[i_mod][2:] = np.maximum.accumulate(path_corrs_gap_no_col_skip + \
                                                      GAP_EXTEND_COST * np.arange(video_spec.shape[0]-2)) - \
                                                      GAP_EXTEND_COST * np.arange(video_spec.shape[0]-2)
    pred_matrix[1][i][2:][path_corrs_gap[i_mod][2:] > path_corrs_gap_no_col_skip] = 7
    path_corrs_gap[i_mod][2:] -= GAP_EXTEND_COST
  
  # reconstruct optimal path by following predecessors backwards through the table
  end_node_layer = np.argmax([path_corrs_match[i_mod,-1],
                              path_corrs_gap[  i_mod,-1]])
  cur_node = (end_node_layer, audio_desc_spec.shape[0]-1, video_spec.shape[0]-1)
  get_predecessor = lambda node: pred_map[pred_matrix[node]](node)
  path = []
  visited = set()
  while min(cur_node[1:]) >= 0:
    cur_node, last_node = get_predecessor(cur_node), cur_node
    # failsafe to prevent an infinite loop that should never happen anyways
    if cur_node in visited:
      break
    visited.add(cur_node)
    if last_node[0] == 0:
      path.append(last_node[1:])
  path = path[::-1]
  
  # determine how much information this node gives about the alignment
  # a larger double derivative means more precise timing information
  # sudden noises give more timing information than droning sounds
  def get_fisher_info(node):
    i,j = node
    if node[0] >= audio_desc_spec.shape[0]-1 or \
       node[1] >= video_spec.shape[0]-1 or \
       min(node) <= 0:
      return 0
    info = 2*np.dot(audio_desc_spec[i  ],video_spec[j  ]) - \
             np.dot(audio_desc_spec[i-1],video_spec[j+1]) - \
             np.dot(audio_desc_spec[i+1],video_spec[j-1])
    info /= min(.2, TIMESTEP_SIZE_SECONDS)
    return info
  
  # the quality of a node combines the correlation of its tokens
  # with how precisely the match is localized in time
  def get_match_quality(node):
    # correlations are between -1 and 1, as all tokens have unit norm
    token_correlation = np.dot(audio_desc_spec[node[0]],video_spec[node[1]])
    fisher_info = min(max(0, get_fisher_info(node)), 10)
    return max(0, token_correlation - MIN_CORR_FOR_TOKEN_MATCH) * (fisher_info / 5)
  
  # filter out low match quality nodes from LCS path
  quals = [get_match_quality(node) for node in path]
  if len(quals) == 0 or max(quals) <= 0:
    raise RuntimeError("Rough alignment failed, are the input files mismatched?")
  path, quals = zip(*[(path, qual) for (path, qual) in zip(path, quals) if qual > 0])
  
  # convert units of path nodes from timesteps to seconds
  path = [(audio_desc_timings[i], video_timings[j]) for (i,j) in path]
  
  return path, quals

# chunk path segments of similar slope into clips
# a clip has the form: (start_index, end_index)
def chunk_path(smooth_path, tol):
  x,y = zip(*smooth_path)
  slopes = np.diff(y) / np.diff(x)
  median_slope = np.median(slopes)
  slope_changes = np.diff(slopes)
  breaks = np.where(np.abs(slope_changes) > tol)[0] + 1
  breaks = [0] + list(breaks) + [len(x)-1]
  clips = list(zip(breaks[:-1], breaks[1:]))
  return clips, median_slope, slopes

# find piece-wise linear alignment that minimizes the weighted combination of
# total absolute error at each node and total absolute slope change of the fit
# distance between nodes and the fit (i.e. errors) are weighted by node quality
# absolute slope changes are differences between the slopes of adjacent fit lines
# slope changes are weighted much more than node errors to smooth out noise
# the main source of noise is rough alignment drift while the describer is speaking
def smooth_align(path, quals, smoothness):
  # rotate basis to make vertical and horizontal slopes "cost" the same
  # the new horizontal axis is x+y and the new vertical is -x+y
  # Wagner–Fischer gives monotonically increasing nodes, so 0 <= slope < inf
  # after this transformation, we instead have -1 <= slope < 1
  # perfectly matching audio has pre-transformation slope = 1
  # after this transformation, it instead has slope = 0
  rotated_path = [(x+y,-x+y) for x,y in path]
  
  # stretch the x axis to make all slopes "cost" nearly the same
  # without this, small changes to the slope at slope = +/-1
  # cost sqrt(2) times as much as small changes at slope = 0
  # by stretching, we limit the range of slopes to within +/- 1/x_stretch_factor
  # the small angle approximation means these slopes all cost roughly the same
  x_stretch_factor = 10.
  rotated_stretched_path = [(x_stretch_factor*x,y) for x,y in rotated_path]
  
  # L1-Minimization to solve the alignment problem using a linear program
  # the absolute value functions needed for "absolute error" can be represented
  # in a linear program by splitting variables into positive and negative pieces
  # and constraining each to be positive (done by default in scipy's linprog)
  # x is fit_err_pos, fit_err_neg, slope_change_pos, slope_change_neg
  # fit_err[i] = path[i][1] - y_fit[i]
  # slope_change[i] = (y_fit[i+2] - y_fit[i+1])/(path[i+2][0] - path[i+1][0]) - \
  #                   (y_fit[i+1] - y_fit[i  ])/(path[i+1][0] - path[i  ][0])
  # this can be rewritten in terms of fit_err by re-arranging the 1st equation:
  #   y_fit[i] = path[i][1] - fit_err[i]
  # this gives:
  #   slope_change[i] = path_half[i] - fit_err_half[i]
  #   where each half is just the original equation but y_fit is swapped out
  # the slope_change variables can then be set using equality constraints
  num_fit_points = len(rotated_stretched_path)
  x,y = [np.array(arr) for arr in zip(*rotated_stretched_path)]
  x_diffs = np.diff(x, prepend=[-10**10], append=[10**10])
  y_diffs = np.diff(y, prepend=[  0    ], append=[ 0    ])
  slope_change_magnitudes = np.abs(np.diff(y_diffs/x_diffs)) * x_stretch_factor
  slope_change_locations = (slope_change_magnitudes > MAX_RATE_RATIO_DIFF_ALIGN)
  slope_change_locations[1:-1] *= (np.abs(y[2:] - y[:-2]) > 5)
  slope_change_costs = np.full(num_fit_points, smoothness / float(TIMESTEP_SIZE_SECONDS))
  slope_change_costs[slope_change_locations] /= PREF_CUT_AT_GAPS_FACTOR
  c = np.hstack([quals,
                 quals,
                 slope_change_costs * x_stretch_factor,
                 slope_change_costs * x_stretch_factor])
  fit_err_coeffs = scipy.sparse.diags([ 1. / x_diffs[:-1],
                                       -1. / x_diffs[:-1] - 1. / x_diffs[1:],
                                                            1. / x_diffs[1:]],
                                      offsets=[0,1,2],
                                      shape=(num_fit_points, num_fit_points + 2)).tocsc()[:,1:-1]
  A_eq = scipy.sparse.hstack([ fit_err_coeffs,
                              -fit_err_coeffs,
                               scipy.sparse.eye(num_fit_points),
                              -scipy.sparse.eye(num_fit_points)])
  b_eq = y_diffs[1:  ] / x_diffs[1:  ] - \
         y_diffs[ :-1] / x_diffs[ :-1]
  fit = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs-ds')
  # if dual simplex solver encounters numerical problems, retry with interior point solver
  if not fit.success and fit.status == 4:
    fit = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs-ipm')
  if not fit.success:
    print(fit)
    raise RuntimeError("Smooth Alignment L1-Min Optimization Failed!")
  
  # combine fit_err_pos and fit_err_neg
  fit_err = fit.x[:num_fit_points] - fit.x[num_fit_points:2*num_fit_points]
  
  # subtract fit errors from nodes to retrieve the smooth fit's coordinates
  # also, unstretch x axis and rotate basis back, reversing the affine pre-processing
  smooth_path = [(((x / x_stretch_factor) - y) / 2.,
                  ((x / x_stretch_factor) + y) / 2.) for x,y in zip(x, y - fit_err)]
  
  # clip off start/end of replacement audio if it doesn't match or isn't aligned
  # without this, describer intro/outro skips can cause mismatches at the start/end
  # the problem would be localized and just means audio might not match video at the start/end
  # instead we just keep the original video's audio in those segments if mismatches are detected
  # if instead the first few or last few nodes are well-aligned, that edge is marked as synced
  # during audio replacement, synced edges will be extended backwards/forwards as far as possible
  # this is useful when the describer begins talking immediately (or before any alignable audio)
  # or when the describer continues speaking until the end (or no more alignable audio remains)
  # otherwise, the mismatch would result in the describer's voice not replacing audio in that part
  max_sync_err = MAX_START_END_SYNC_ERR_SECONDS
  smoothing_std = MIN_START_END_SYNC_TIME_SECONDS / (2. * TIMESTEP_SIZE_SECONDS)
  smoothed_fit_err = nd.gaussian_filter(np.abs(fit_err), sigma=smoothing_std)
  smooth_err_path = zip(smoothed_fit_err, smooth_path)
  old_length = num_fit_points
  smooth_err_path = list(itertools.dropwhile(lambda x: x[0] > max_sync_err, smooth_err_path))[::-1]
  is_synced_at_start = len(smooth_err_path) == old_length
  old_length = len(smooth_err_path)
  smooth_err_path = list(itertools.dropwhile(lambda x: x[0] > max_sync_err, smooth_err_path))[::-1]
  is_synced_at_end = len(smooth_err_path) == old_length
  _, smooth_path = zip(*smooth_err_path)
  smooth_path = list(smooth_path)
  if is_synced_at_start:
    slope = (smooth_path[1][1] - smooth_path[0][1]) / (smooth_path[1][0] - smooth_path[0][0])
    smooth_path.insert(0, (-10e10, -10e10 * slope))
  if is_synced_at_end:
    slope = (smooth_path[-1][1] - smooth_path[-2][1]) / (smooth_path[-1][0] - smooth_path[-2][0])
    smooth_path.append((10e10, 10e10 * slope))
  
  clips, median_slope, slopes = chunk_path(smooth_path, tol=1e-7)
  
  # assemble clips with slopes within the rate tolerance into runs
  runs, run = [], []
  bad_clips = []
  for clip in clips:
    if np.abs(median_slope-slopes[clip[0]]) > MAX_RATE_RATIO_DIFF_ALIGN:
      if len(run) > 0:
        runs.append(run)
        run = []
      bad_clips.append(clip)
      continue
    run.append(clip)
  if len(run) > 0:
    runs.append(run)
  
  return smooth_path, runs, bad_clips, clips

# if the start or end were marked as synced during smooth alignment then
# extend that alignment to the edge (i.e. to the start/end of the audio)
def cap_synced_end_points(smooth_path, video_arr, audio_desc_arr):
  if smooth_path[0][0] < -10e9:
    slope = smooth_path[0][1] / smooth_path[0][0]
    new_start_point = (0, smooth_path[1][1] - smooth_path[1][0] * slope)
    if new_start_point[1] < 0:
      new_start_point = (smooth_path[1][0] - smooth_path[1][1] / slope, 0)
    smooth_path[0] = new_start_point
  if smooth_path[-1][0] > 10e9:
    video_runtime = (video_arr.shape[1] - 2.) / AUDIO_SAMPLE_RATE
    audio_runtime = (audio_desc_arr.shape[1] - 2.) / AUDIO_SAMPLE_RATE
    slope = smooth_path[-1][1] / smooth_path[-1][0]
    new_end_point = (audio_runtime, smooth_path[-2][1] + (audio_runtime - smooth_path[-2][0]) * slope)
    if new_end_point[1] > video_runtime:
      new_end_point = (smooth_path[-2][0] + (video_runtime - smooth_path[-2][1]) / slope, video_runtime)
    smooth_path[-1] = new_end_point

# visualize both the rough and smooth alignments
def plot_alignment(plot_filename_no_ext, path, smooth_path, quals, runs,
                   bad_clips, ad_timings, similarity_percent):
  scatter_color = [.2,.4,.8]
  lcs_rgba = np.zeros((len(quals),4))
  lcs_rgba[:,:3] = np.array(scatter_color)[None,:]
  lcs_rgba[:,3] = np.minimum(1, np.array(quals) * 500. / len(quals))
  audio_times, video_times = np.array(path).T.reshape((2,-1))
  audio_offsets = audio_times - video_times
  def expand_limits(start, end, ratio=.01):
    average = (end + start) / 2.
    half_diff = (end - start) / 2.
    half_diff *= (1 + ratio)
    return (average - half_diff, average + half_diff)
  plt.xlim(expand_limits(*(0, np.max(video_times) / 60.)))
  plt.ylim(expand_limits(*(np.min(audio_offsets) - TIMESTEP_SIZE_SECONDS / 2.,
                          np.max(audio_offsets) + TIMESTEP_SIZE_SECONDS / 2.)))
  plt.scatter(video_times / 60., audio_offsets, s=3, c=lcs_rgba, label='LCS Matches')
  audio_times, video_times = np.array(smooth_path).T.reshape((2,-1))
  audio_offsets = audio_times - video_times
  if ad_timings is None:
    plt.plot(video_times / 60., audio_offsets, 'r-', lw=.5, label='Replaced Audio')
    bad_path = []
    for clip in bad_clips:
      bad_path.extend(smooth_path[clip[0]:clip[1]+1])
      bad_path.append((smooth_path[clip[1]][0] + 1e-10, np.nan))
    audio_times, video_times = np.array(bad_path).T.reshape((2,-1))
    audio_offsets = audio_times - video_times
    if len(audio_offsets) > 0:
      plt.plot(video_times / 60., audio_offsets, 'c-', lw=1, label='Original Audio')
  else:
    interp = scipy.interpolate.interp1d(video_times, audio_offsets,
                                        fill_value = np.inf,
                                        bounds_error = False, assume_sorted = True)
    plt.plot(video_times / 60., audio_offsets, 'c-', lw=.5, label='Original Audio')
    video_times = ad_timings
    audio_offsets = interp(ad_timings)
    if len(audio_offsets) > 0:
      plt.plot(video_times / 60., audio_offsets, 'r-', lw=1, label='Replaced Audio')
  plt.xlabel('Video Time (minutes)')
  plt.ylabel('Audio Description Offset (seconds)')
  plt.title(f"Alignment - Media Similarity {similarity_percent:.2f}%")
  plt.legend().legend_handles[0].set_color(scatter_color)
  plt.tight_layout()
  plt.savefig(plot_filename_no_ext + '.png', dpi=400)
  plt.clf()
  
  with open(plot_filename_no_ext + '.txt', 'w') as file:
    rough_clips, median_slope, _ = chunk_path(smooth_path, tol=2e-2)
    video_offset = np.diff(smooth_path[rough_clips[0][0]])[0]
    print(f"Input file similarity: {similarity_percent:.2f}%", file=file)
    print("Main changes needed to video to align it to audio input:", file=file)
    print(f"Start Offset: {-video_offset:.2f} seconds", file=file)
    print(f"Median Rate Change: {(median_slope-1.)*100:.2f}%", file=file)
    for clip_start, clip_end in rough_clips:
      audio_desc_start, video_start = smooth_path[clip_start]
      audio_desc_end, video_end = smooth_path[clip_end]
      slope = (video_end - video_start) / (audio_desc_end - audio_desc_start)
      def str_from_time(seconds):
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:2.0f}:{minutes:02.0f}:{seconds:05.2f}"
      print(f"Rate change of {(slope-1.)*100:6.1f}% from {str_from_time(video_start)} to " + \
            f"{str_from_time(video_end)} aligning with audio from " + \
            f"{str_from_time(audio_desc_start)} to {str_from_time(audio_desc_end)}", file=file)

# use the smooth alignment to replace runs of video sound with corresponding described audio
def replace_aligned_segments(video_arr, audio_desc_arr, smooth_path, runs, no_pitch_correction=False):
  # perform quadratic interpolation of the audio description's waveform
  # this allows it to be stretched to match the corresponding video segment
  def audio_desc_arr_interp(samples):
    chunk_size = 10**7
    interpolated_chunks = []
    for chunk in (samples[i:i+chunk_size] for i in range(0, len(samples), chunk_size)):
      interp_bounds = (max(int(chunk[0]-2), 0),
                       min(int(chunk[-1]+2), audio_desc_arr.shape[1]))
      interp = scipy.interpolate.interp1d(np.arange(*interp_bounds),
                                          audio_desc_arr[:,slice(*interp_bounds)],
                                          copy=False, bounds_error=False, fill_value=0,
                                          kind='quadratic', assume_sorted=True)
      interpolated_chunks.append(interp(chunk).astype(np.float32))
    return np.hstack(interpolated_chunks)
  
  # construct a stretched audio description waveform using the quadratic interpolator
  def get_interped_segment(run, interp):
    segment = []
    for clip in run:
      num_samples = int(y[clip[1]] * AUDIO_SAMPLE_RATE) - \
                    int(y[clip[0]] * AUDIO_SAMPLE_RATE)
      clip_bounds = np.array((x[clip[0]], x[clip[1]])) * AUDIO_SAMPLE_RATE
      sample_points = np.linspace(*clip_bounds, num=num_samples, endpoint=False)
      segment.append(interp(sample_points))
    segment = np.hstack(segment)
    return segment
  
  # compress dual channel audio to mono for use in pitch corrected stretching
  # pytsmod's wsola treats channels separately, so without this it sounds weird
  if not no_pitch_correction:
    audio_desc_arr_mono = np.mean(audio_desc_arr, axis=0)
  
  x,y = zip(*smooth_path)
  for run in runs:
    run_length_seconds = y[run[-1][1]] - y[run[0][0]]
    if run_length_seconds < MIN_DURATION_TO_REPLACE_SECONDS:
      continue
    anchor_point_path_indices = [clip[0] for clip in run]
    anchor_point_path_indices.append(run[-1][1])
    anchor_points = (np.array((np.array(x)[anchor_point_path_indices],
                               np.array(y)[anchor_point_path_indices])) * AUDIO_SAMPLE_RATE).astype(int)
    slopes = np.diff(anchor_points[1]) / np.diff(anchor_points[0])
    for clip_index, (clip, slope) in enumerate(zip(run, slopes)):
      # only apply pitch correction if the difference would be noticeable
      if no_pitch_correction or np.abs(1 - slope) <= JUST_NOTICEABLE_DIFF_IN_FREQ_RATIO:
        stretched_audio = get_interped_segment([clip], audio_desc_arr_interp)
      else:
        anchor_point_pair = anchor_points[:,clip_index:clip_index+2].copy()
        # account for quirks of pytsmod's wsola anchor point implementation
        anchor_point_pair[1][-1] -= 1
        anchor_y_offset = anchor_point_pair[1][0]
        anchor_point_pair[1,:] -= anchor_y_offset
        stretched_audio = pytsmod.wsola(audio_desc_arr_mono, anchor_point_pair)
      video_arr[:,slice(*anchor_points[1,clip_index:clip_index+2])] = stretched_audio

# identify which segments of the replaced audio actually have the describer speaking
# uses a Naive Bayes classifier smoothed with L1-Minimization to identify the describer
def detect_describer(video_arr, video_spec, video_spec_raw, video_timings,
                     smooth_path, detect_sensitivity, boost_sensitivity):
  # retokenize the audio description, which has been stretched to match the video
  audio_desc_spec_raw, audio_timings = tokenize_audio(video_arr)
  audio_desc_spec = normalize_spec(audio_desc_spec_raw)
  
  # avoid boosting or training on mismatched segments, like those close to skips
  # assumes matching segments all have the same, constant play rate
  # could be modified to handle a multi-modal distribution of rates
  aligned_audio_times, aligned_video_times = zip(*smooth_path)
  interp = scipy.interpolate.interp1d(aligned_video_times, aligned_audio_times,
                                      fill_value = 'extrapolate',
                                      bounds_error = False, assume_sorted = True)
  slopes = (interp(video_timings + 1e-5) - \
            interp(video_timings - 1e-5)) / 2e-5
  median_slope = np.median(slopes)
  aligned_mask =      np.abs(slopes - median_slope) < MAX_RATE_RATIO_DIFF_ALIGN
  well_aligned_mask = np.abs(slopes - median_slope) < MAX_RATE_RATIO_DIFF_BOOST
  
  # first pass identification by assuming poorly matched tokens are describer speech
  # also assumes the describer doesn't speak very quietly
  corrs = np.sum(audio_desc_spec * video_spec, axis=-1)
  smooth_volume = nd.gaussian_filter(audio_desc_spec[:,0], sigma=1)
  audio_desc_loud = smooth_volume > np.percentile(smooth_volume, 30)
  speech_mask = (corrs < .2) * audio_desc_loud
  
  # normalize spectrogram coefficients along time axis to prep for conversion to PDFs
  audio_desc_spec = normalize_spec(audio_desc_spec_raw, axes=(0,))
  audio_desc_spec = np.clip(audio_desc_spec / 6., -1, 1)
  video_spec = normalize_spec(video_spec_raw, axes=(0,))
  video_spec = np.clip(video_spec / 6., -1, 1)
  
  # convert sampled features (e.g. spectrogram) to probability densities of each feature
  # when given a spectrogram, finds the distributions of the MFC coefficients
  def make_log_pdfs(arr):
    resolution = 100
    bins_per_spot = 4
    num_bins = int(resolution * bins_per_spot)
    uniform_prior_strength_per_spot = 1
    uniform_prior_strength_per_bin = uniform_prior_strength_per_spot / float(bins_per_spot)
    bin_range = (-1 - 1e-10, 1 + 1e-10)
    get_hist = lambda x: np.histogram(x, bins=num_bins, range=bin_range)[0]
    pdfs = np.apply_along_axis(get_hist, 1, arr.T)
    pdfs = pdfs + uniform_prior_strength_per_bin
    smooth = lambda x: nd.gaussian_filter(x, sigma=bins_per_spot)
    pdfs = np.apply_along_axis(smooth, 1, pdfs)
    pdfs = pdfs / np.sum(pdfs[0,:])
    log_pdfs = np.log(pdfs)
    bin_edges = np.histogram([], bins=num_bins, range=bin_range)[1]
    return log_pdfs, bin_edges
  
  diff_spec = audio_desc_spec - video_spec
  diff_spec = np.clip(diff_spec, -1, 1)
  
  # Naive Bayes classifier to roughly estimate whether each token is describer speech
  desc_log_pdfs, _ = make_log_pdfs(diff_spec[speech_mask * well_aligned_mask])
  nondesc_log_pdfs, bin_edges = make_log_pdfs(diff_spec[(~speech_mask) * well_aligned_mask])
  lratio_lookup = desc_log_pdfs - nondesc_log_pdfs
  lratios = lratio_lookup[np.fromfunction(lambda i,j: j, diff_spec.shape, dtype=int),
                          np.digitize(diff_spec, bin_edges, right=True)-1]
  ratio_desc_to_nondesc = np.sum(speech_mask * well_aligned_mask) /\
                         (np.sum((~speech_mask) * well_aligned_mask) + 1.)
  relative_probs = np.sum(lratios, axis=1)
  relative_probs /= np.std(relative_probs)
  relative_probs -= np.mean(relative_probs)
  
  # L1-Minimization to smoothly identify audio descriptions using a linear program
  # x is fit_err_pos, fit_err_neg, delta_fit_pos, delta_fit_neg
  # fit_err[i] = relative_probs[i] - y_fit[i]
  # delta_fit[i] = y_fit[i] - y_fit[i-1]
  # this can be rewritten in terms of fit_err by re-arranging the 1st equation:
  #   y_fit[i] = relative_probs[i] - fit_err[i]
  # this gives:
  #   delta_fit[i] = (relative_probs[i] - relative_probs[i-1]) -\
  #                  (fit_err[i] - fit_err[i-1])
  # the delta_fit variables can then be set using equality constraints
  num_fit_points = len(relative_probs)
  y_diffs = np.diff(relative_probs)
  pos_err_cost_factor = MIN_DESC_DURATION / float(TIMESTEP_SIZE_SECONDS)
  neg_err_cost_factor = MAX_GAP_IN_DESC_SEC / float(TIMESTEP_SIZE_SECONDS)
  c = np.hstack([np.ones(num_fit_points) / pos_err_cost_factor,
                 np.ones(num_fit_points) / neg_err_cost_factor,
                 np.ones(num_fit_points - 1) / 2.,
                 np.ones(num_fit_points - 1) / 2.])
  fit_err_coeffs = scipy.sparse.diags([-np.ones(num_fit_points),
                                        np.ones(num_fit_points)],
                                      offsets=[0,1],
                                      shape=(num_fit_points - 1, num_fit_points)).tocsc()
  A_eq = scipy.sparse.hstack([ fit_err_coeffs,
                              -fit_err_coeffs,
                               scipy.sparse.eye(num_fit_points-1),
                              -scipy.sparse.eye(num_fit_points-1)])
  b_eq = y_diffs
  fit = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs-ds')
  # if dual simplex solver encounters numerical problems, retry with interior point solver
  if not fit.success and fit.status == 4:
    fit = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs-ipm')
  if not fit.success:
    print(fit)
    raise RuntimeError("Describer Voice Detection L1-Min Optimization Failed!")
  
  # combine fit_err_pos and fit_err_neg
  fit_err = fit.x[:num_fit_points] - fit.x[num_fit_points:2*num_fit_points]
  
  # subtract fit errors from nodes to retrieve the smoothed fit
  smooth_desc_locations = relative_probs - fit_err
  
  # hard threshold to classify each token as describer speech or not
  speech_mask = smooth_desc_locations > 1. - 1.5 * detect_sensitivity
  speech_mask *= aligned_mask
  
  # a separate mask is created for describer volume boosting
  # as losing the describer's voice entirely is usually worse than it just being quiet
  # and imperfectly aligned segments may have descriptions, but shouldn't be boosted
  boost_mask = smooth_desc_locations > 1. - 1.5 * boost_sensitivity
  boost_mask *= well_aligned_mask
  
  # convert a token classification into a mask that can be applied directly to samples
  # unlike the input, the output isn't a boolean array but an array of floats
  def token_mask_to_sample_mask(token_mask):
    description_timings = video_timings[1:-1][token_mask[1:-1]]
    sample_mask = np.zeros(video_arr.shape[1], dtype=np.float32)
    window_radius = int(AUDIO_SAMPLE_RATE * TIMESTEP_SIZE_SECONDS)
    window_size_seconds = 2 * window_radius + 1
    bump = scipy.signal.windows.hann(window_size_seconds)
    for description_timing in description_timings:
      window_center = int(description_timing * AUDIO_SAMPLE_RATE)
      sample_mask[window_center-window_radius:window_center+window_radius+1] += bump
    return sample_mask
  
  speech_sample_mask = token_mask_to_sample_mask(speech_mask)
  boost_sample_mask = token_mask_to_sample_mask(boost_mask)
  ad_timings = video_timings.copy()
  ad_timings[~speech_mask] = np.inf
  
  return speech_sample_mask, boost_sample_mask, ad_timings

# Convert piece-wise linear fit to ffmpeg expression for editing video frame timestamps
def encode_fit_as_ffmpeg_expr(smooth_path, clips, video_offset, start_key_frame):
  # PTS is the input frame's presentation timestamp, which is when frames are displayed
  # TB is the timebase, which is how many seconds each unit of PTS corresponds to
  # the output value of the expression will be the frame's new PTS
  setts_cmd = ['TS']
  start_skip = max(0, video_offset - start_key_frame)
  if start_skip > 0:
    # lossless cutting can only happen at key frames, so we cut the video before the audio starts
    # but that means the video is behind the audio and needs to catch up by playing quicker
    # catchup_spread is the ratio of time to spend catching up to the amount of catching up needed
    catchup_spread = 1./CATCHUP_RATE
    setts_cmd.append(f'+clip(TS-{start_key_frame},0,{start_skip*(1+catchup_spread)}/TB)*{-1./(1+catchup_spread)}')
  elif video_offset < 0:
    # if the audio starts before the video, stretch the first frame of the video back to meet it
    setts_cmd.append(f'+clip(TS-{start_key_frame},0,{-video_offset/10000.}/TB)*10000')
  # each segment of the linear fit can be encoded as a single clip function
  setts_cmd.append('+(0')
  for clip_start, clip_end in clips:
    audio_desc_start, video_start = smooth_path[clip_start]
    audio_desc_end, video_end = smooth_path[clip_end]
    video_start -= start_key_frame
    video_end -= start_key_frame
    audio_desc_length = audio_desc_end - audio_desc_start
    video_length = video_end - video_start
    slope = audio_desc_length / video_length
    setts_cmd.append(f'+clip(TS-{start_key_frame}-{video_start:.4f}/TB,0,{max(0,video_length):.4f}/TB)*{slope-1:.9f}')
  setts_cmd.append(')')
  setts_cmd = ''.join(setts_cmd)
  return setts_cmd

def get_ffmpeg():
  return static_ffmpeg.run._get_or_fetch_platform_executables_else_raise_no_lock()[0]

def get_ffprobe():
  return static_ffmpeg.run._get_or_fetch_platform_executables_else_raise_no_lock()[1]

def get_closest_key_frame_time(video_file, time):
  if time <= 0:
    return 0
  key_frames = ffmpeg.probe(video_file, cmd=get_ffprobe(), select_streams='V',
                            show_frames=None, skip_frame='nokey')['frames']
  key_frame_times = np.array([float(frame['pts_time']) for frame in key_frames] + [0])
  return np.max(key_frame_times[key_frame_times <= time])

# outputs a new media file with the replaced audio (which includes audio descriptions)
def write_replaced_media_to_disk(output_filename, media_arr, video_file=None, audio_desc_file=None,
                                 setts_cmd=None, start_key_frame=None):
  if audio_desc_file is None:
    media_input = ffmpeg.input('pipe:', format='s16le', acodec='pcm_s16le',
                               ac=2, ar=AUDIO_SAMPLE_RATE)
    if video_file is None or os.path.splitext(output_filename)[1][1:] in AUDIO_EXTENSIONS:
      write_command = ffmpeg.output(media_input, output_filename, loglevel='fatal').overwrite_output()
    else:
      original_video = ffmpeg.input(video_file)
      # "-max_interleave_delta 0" is sometimes necessary to fix an .mkv bug that freezes audio/video:
      #   ffmpeg bug warning: [matroska @ 0000000002c814c0] Starting new cluster due to timestamp
      # more info about the bug and fix: https://reddit.com/r/ffmpeg/comments/efddfs/
      write_command = ffmpeg.output(media_input, original_video, output_filename,
                                    acodec='copy', vcodec='copy', scodec='copy',
                                    max_interleave_delta='0', loglevel='fatal',
                                    **{"c:a:0": "aac", "disposition:a:0": "default"}).overwrite_output()
    ffmpeg_caller = write_command.run_async(pipe_stdin=True, cmd=get_ffmpeg())
    ffmpeg_caller.stdin.write(media_arr.astype(np.int16).T.tobytes())
    ffmpeg_caller.stdin.close()
    ffmpeg_caller.wait()
  else:
    media_input = ffmpeg.input(audio_desc_file)
    audio_desc_streams = ffmpeg.probe(audio_desc_file, cmd=get_ffprobe(), select_streams='a',
                                      show_entries='format=duration')['streams']
    audio_desc_duration = max([float(stream['duration']) for stream in audio_desc_streams])
    original_video = ffmpeg.input(video_file, an=None, ss=start_key_frame)
    if os.path.splitext(output_filename)[1] == os.path.splitext(video_file)[1]:
      # wav files don't have codecs compatible with most video containers, so we convert to aac
      audio_codec = 'copy' if os.path.splitext(audio_desc_file)[1] != '.wav' else 'aac'
      # flac audio may only have experimental support in some video containers (e.g. mp4)
      standards = 'normal' if os.path.splitext(audio_desc_file)[1] != '.flac' else 'experimental'
      write_command = ffmpeg.output(media_input, original_video, output_filename,
                                    acodec=audio_codec, vcodec='copy', scodec='copy',
                                    max_interleave_delta='0', loglevel='fatal', strict=standards,
                                    **{'bsf:v': f'setts=ts=\'{setts_cmd}\'',
                                       'bsf:s': f'setts=ts=\'{setts_cmd}\''}).overwrite_output()
      write_command.run(cmd=get_ffmpeg())
    else:
      # work around for bug that sometimes breaks setts when output and input formats differ
      # the trick is separating the input and output by piping from one ffmpeg process into another
      # mkv files break if 'nut' is used, while other files break when 'matroska' is used
      format = 'matroska' if os.path.splitext(output_filename)[1] == '.mkv' else 'nut'
      write_command = ffmpeg.output(original_video, 'pipe:', format=format, vsync='passthrough',
                                    c='copy', loglevel='fatal')
      ffmpeg_caller = write_command.run_async(pipe_stdout=True, cmd=get_ffmpeg())
      pipe_input = ffmpeg.input('pipe:', format=format, thread_queue_size='512')
      write_command2 = ffmpeg.output(media_input, pipe_input, output_filename, c='copy',
                                     max_interleave_delta='0', loglevel='fatal', vsync='passthrough',
                                     **{'bsf:v': f'setts=ts=\'{setts_cmd}\'',
                                        'bsf:s': f'setts=ts=\'{setts_cmd}\''}).overwrite_output()
      ffmpeg_caller2 = write_command2.run_async(pipe_stdin=True, cmd=get_ffmpeg())
      while True:
        in_bytes = ffmpeg_caller.stdout.read(100000)
        if not in_bytes:
          break
        ffmpeg_caller2.stdin.write(in_bytes)
      ffmpeg_caller2.stdin.close()
      ffmpeg_caller.wait()
      ffmpeg_caller2.wait()


# check whether static_ffmpeg has already installed ffmpeg and ffprobe
def is_ffmpeg_installed():
  ffmpeg_dir = static_ffmpeg.run.get_platform_dir()
  indicator_file = os.path.join(ffmpeg_dir, "installed.crumb")
  return os.path.exists(indicator_file)

# combines videos with matching audio files (e.g. audio descriptions)
# this is the main function of this script, it calls the other functions in order
def combine(video, audio, smoothness=50, stretch_audio=False, keep_non_ad=False,
            boost=0, ad_detect_sensitivity=.6, boost_sensitivity=.4, yes=False,
            prepend="ad_", no_pitch_correction=False, output_dir=default_output_dir,
            alignment_dir=default_alignment_dir, extension="copy", display_func=None):
  video_files, video_file_types = get_sorted_filenames(video, VIDEO_EXTENSIONS, AUDIO_EXTENSIONS)
  
  if yes == False and sum(video_file_types) > 0:
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
  
  display("", display_func)
  ensure_folders_exist([output_dir], display_func)
  if PLOT_ALIGNMENT_TO_FILE:
    ensure_folders_exist([alignment_dir], display_func)
  
  display("", display_func)
  for (video_file, audio_desc_file) in zip(video_files, audio_desc_files):
    display(os.path.split(video_file)[1], display_func)
    display(os.path.split(audio_desc_file)[1], display_func)
    display("", display_func)
  if yes == False:
    print("Are the above input file pairings correct?")
    print("If not, press ctrl+c to kill this script.")
    input("If they are correct, press Enter to continue...")
    print("")
  
  # if ffmpeg isn't installed, install it
  if not is_ffmpeg_installed():
    display("Downloading and installing ffmpeg (media editor, 50 MB download)...", display_func)
    get_ffmpeg()
    if not is_ffmpeg_installed():
      RuntimeError("Failed to install ffmpeg.")
    display("Successfully installed ffmpeg.", display_func)
  
  display("Processing files:", display_func)
  
  for (video_file, audio_desc_file, video_filetype) in zip(video_files, audio_desc_files,
                                                           video_file_types):
    # Default is to use the input video's extension for the output video
    if extension is None or extension in ["", "copy"]:
      ext = os.path.splitext(video_file)[1]
    else:
      # add a dot to the extension if it's missing
      ext = ('' if extension[0] == '.' else '.') + extension
    output_filename = prepend + os.path.splitext(os.path.split(video_file)[1])[0] + ext
    output_filename = os.path.join(output_dir, output_filename)
    display(f" {output_filename}", display_func)
    
    if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
      display("   output file already exists, skipping...", display_func)
      continue
    
    # print warning if output file's full path is longer than Windows MAX_PATH (260)
    full_output_filename = os.path.abspath(output_filename)
    if IS_RUNNING_WINDOWS and len(full_output_filename) >= 260:
      display("   WARNING: very long output path, ffmpeg may fail...", display_func)
    
    video_arr = parse_audio_from_file(video_file)
    audio_desc_arr = parse_audio_from_file(audio_desc_file)
    video_spec_raw, video_timings = tokenize_audio(video_arr)
    video_spec = normalize_spec(video_spec_raw)
    audio_desc_spec_raw, audio_desc_timings = tokenize_audio_dither(audio_desc_arr, video_timings)
    audio_desc_spec = normalize_spec(audio_desc_spec_raw)
    
    # rescale RMS intensity of audio to match video
    audio_desc_arr *= (np.std(video_arr) / np.std(audio_desc_arr))
    
    path, quals = rough_align(video_spec, audio_desc_spec, video_timings, audio_desc_timings)
    
    similarity_ratio = float(len(quals)) / max(video_spec.shape[0], audio_desc_spec.shape[0])
    similarity_percent = min(100, 100 * similarity_ratio)
    if similarity_percent < 10:
      display(f"   WARNING: similarity {similarity_percent:.1f}%, likely mismatched files", display_func)
    if similarity_percent > 90:
      display(f"   WARNING: similarity {similarity_percent:.1f}%, likely undescribed media", display_func)
    
    smooth_path, runs, bad_clips, clips = smooth_align(path, quals, smoothness)
    
    cap_synced_end_points(smooth_path, video_arr, audio_desc_arr)
    
    ad_timings = None
    if stretch_audio:
      if keep_non_ad:
        video_arr_original = video_arr.copy()
      
      replace_aligned_segments(video_arr, audio_desc_arr, smooth_path, runs, no_pitch_correction)
      del audio_desc_arr
      
      if keep_non_ad or boost != 0:
        outputs = detect_describer(video_arr, video_spec, video_spec_raw, video_timings,
                                   smooth_path, ad_detect_sensitivity, boost_sensitivity)
        speech_sample_mask, boost_sample_mask, ad_timings = outputs
      if keep_non_ad:
        video_arr *= speech_sample_mask
        video_arr += video_arr_original * (1 - speech_sample_mask)
        del video_arr_original
        del speech_sample_mask
      else:
        ad_timings = None
      if boost != 0:
        video_arr = video_arr * (1. + (10**(boost / 10.) - 1.) * boost_sample_mask)
        del boost_sample_mask
      
      # prevent peaking by rescaling to within +/- 16,382
      video_arr *= (2**15 - 2.) / np.max(np.abs(video_arr))
      
      if video_filetype == 0:
        write_replaced_media_to_disk(output_filename, video_arr, video_file)
      else:
        write_replaced_media_to_disk(output_filename, video_arr)
    else:
      if video_filetype == 1:
        raise RuntimeError("Argument --stretch_audio is required when both inputs are audio files.")
      if os.path.splitext(output_filename)[1][1:] in AUDIO_EXTENSIONS:
        raise RuntimeError("Argument --stretch_audio is required when output file extension is an audio filetype.")
      video_offset = np.diff(smooth_path[clips[0][0]])[0]
      start_key_frame = get_closest_key_frame_time(video_file, video_offset)
      setts_cmd = encode_fit_as_ffmpeg_expr(smooth_path, clips, video_offset, start_key_frame)
      write_replaced_media_to_disk(output_filename, None, video_file, audio_desc_file,
                                   setts_cmd, start_key_frame)
    
    del video_arr
    if PLOT_ALIGNMENT_TO_FILE:
      plot_filename_no_ext = os.path.join(alignment_dir, os.path.splitext(os.path.split(video_file)[1])[0])
      plot_alignment(plot_filename_no_ext, path, smooth_path, quals,
                     runs, bad_clips, ad_timings, similarity_percent)
  display("All files processed.", display_func)

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
  settings = {'smoothness':           config.getfloat('alignment', 'smoothness', fallback=50),
              'stretch_audio':        config.getboolean('alignment', 'stretch_audio', fallback=False),
              'keep_non_ad':          config.getboolean('alignment', 'keep_non_ad', fallback=False),
              'boost':                config.getfloat('alignment', 'boost', fallback=0),
              'ad_detect_sensitivity':config.getfloat('alignment', 'ad_detect_sensitivity', fallback=.6),
              'boost_sensitivity':    config.getfloat('alignment', 'boost_sensitivity', fallback=.4),
              'prepend':              config.get('alignment', 'prepend', fallback='ad_'),
              'no_pitch_correction':  config.getboolean('alignment', 'no_pitch_correction', fallback=False),
              'output_dir':           config.get('alignment', 'output_dir', fallback=default_output_dir),
              'alignment_dir':        config.get('alignment', 'alignment_dir', fallback=default_alignment_dir),
              'extension':            config.get('alignment', 'extension', fallback='copy')}
  if not config.has_section('alignment'):
    write_config_file(config_path, settings)
  return settings

def settings_gui(config_path: Path):
  settings = read_config_file(config_path)
  layout = [[sg.Text('Check tooltips (i.e. mouse-over text) for descriptions:')],
            [sg.Column([[sg.Text('extension:', size=(10, 1.2), pad=(1,5)),
                         sg.Input(default_text=str(settings['extension']), size=(8, 1.2), pad=(10,5), key='extension',
                                  tooltip='File type of output video (e.g. mkv). When set to "copy", copies the ' + \
                                          'file type of the corresponding input video. Default is "copy".')]])],
            [sg.Column([[sg.Text('prepend:', size=(8, 1.2), pad=(1,5)),
                         sg.Input(default_text=str(settings['prepend']), size=(8, 1.2), pad=(10,5), key='prepend',
                                  tooltip='Output file name prepend text. Default is "ad_"')]])],
            [sg.Column([[sg.Text('output_dir:', size=(10, 1.2), pad=(1,5)),
                         sg.Input(default_text=str(settings['output_dir']), size=(22, 1.2), pad=(10,5), key='output_dir',
                                  tooltip='Directory combined output media is saved to. Default is "videos_with_ad"'),
                                  sg.FolderBrowse(button_text="Browse Folder", key='output_browse')]])],
            [sg.Column([[sg.Text('alignment_dir:', size=(13, 1.2), pad=(1,5)),
                         sg.Input(default_text=str(settings['alignment_dir']), size=(22, 1.2), pad=(10,5), key='alignment_dir',
                                  tooltip='Directory alignment data and plots are saved to. Default is "alignment_plots"'),
                         sg.FolderBrowse(button_text="Browse Folder", key='alignment_browse')]], pad=(2,7))],
            [sg.Column([[sg.Text('smoothness:', size=(12, 1), pad=(1,5)),
                         sg.Input(default_text=str(settings['smoothness']), size=(8, 1.2), pad=(10,5), key='smoothness',
                                  tooltip='Lower values make the alignment more accurate when there are skips ' + \
                                          '(e.g. describer pauses), but also make it more likely to misalign. ' + \
                                          'Default is 50.')]])],
            [sg.Checkbox('stretch_audio', default=settings['stretch_audio'], key='stretch_audio', change_submits=True,
                         tooltip='Stretches the input audio to fit the input video. ' + \
                                 'Default is to stretch the video to fit the audio.')],
            [sg.Checkbox('keep_non_ad', default=settings['keep_non_ad'], key='keep_non_ad',
                         disabled=not settings['stretch_audio'],
                         tooltip='Tries to only replace segments with audio description. Useful if ' + \
                                 'video\'s audio quality is better. Default is to replace all aligned audio. ' + \
                                 'Requires --stretch_audio to be set, otherwise does nothing.')],
            [sg.Column([[sg.Text('boost:', size=(6, 1), pad=(1,5)),
                         sg.Input(default_text=str(settings['boost']), size=(8, 1.2), pad=(10,5),
                                  key='boost', disabled=not settings['stretch_audio'],
                                  tooltip='Boost (or quieten) description volume. Units are decibels (dB), so ' + \
                                          '-3 makes the describer about 2x quieter, while 3 makes them 2x louder. ' + \
                                          'Requires --stretch_audio to be set, otherwise does nothing.')]])],
            [sg.Column([[sg.Text('ad_detect_sensitivity:', size=(21, 1.2), pad=(2,5)),
                         sg.Input(default_text=str(settings['ad_detect_sensitivity']), size=(8, 1.2), pad=(10,5),
                                  key='ad_detect_sensitivity', disabled=not settings['stretch_audio'],
                                  tooltip='Audio description detection sensitivity ratio. Higher values make ' + \
                                          '--keep_non_ad more likely to replace aligned audio. Default is 0.6')]])],
            [sg.Column([[sg.Text('boost_sensitivity:', size=(17, 1.2), pad=(1,5)),
                         sg.Input(default_text=str(settings['boost_sensitivity']), size=(8, 1.2), pad=(10,5),
                                  key='boost_sensitivity', disabled=not settings['stretch_audio'],
                                  tooltip='Higher values make --boost less likely to miss a description, but ' + \
                                          'also make it more likely to boost non-description audio. Default is 0.4')]])],
            [sg.Checkbox('no_pitch_correction', default=settings['no_pitch_correction'], key='no_pitch_correction',
                         disabled=not settings['stretch_audio'],
                         tooltip='Skips pitch correction step when stretching audio. ' + \
                                 'Requires --stretch_audio to be set, otherwise does nothing.')],
            [sg.Column([[sg.Submit('Save', pad=(40,3)),
                         sg.Button('Cancel')]], pad=((135,3),10))]]
  settings_window = sg.Window('Settings - describealign', layout, font=('Arial', 16), finalize=True)
  settings_window['extension'].set_focus()
  while True:
    event, values = settings_window.read()
    if event in (sg.WIN_CLOSED, 'Cancel') or settings_window.TKrootDestroyed:
      break
    if event == 'stretch_audio':
      # work around bug in PySimpleGUIWx's InputText Update function where enabling/disabling are flipped
      if IS_RUNNING_WINDOWS:
        settings_window['boost'].Update(disabled = values['stretch_audio'])
        settings_window['ad_detect_sensitivity'].Update(disabled = values['stretch_audio'])
        settings_window['boost_sensitivity'].Update(disabled = values['stretch_audio'])
      else:
        settings_window['boost'].Update(disabled = not values['stretch_audio'])
        settings_window['ad_detect_sensitivity'].Update(disabled = not values['stretch_audio'])
        settings_window['boost_sensitivity'].Update(disabled = not values['stretch_audio'])
      settings_window['keep_non_ad'].Update(disabled = not values['stretch_audio'])
      settings_window['no_pitch_correction'].Update(disabled = not values['stretch_audio'])
    if event == 'Save':
      settings = values.copy()
      del settings['output_browse']
      del settings['alignment_browse']
      write_config_file(config_path, settings)
      break
  settings_window.close()

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
      traceback.print_exc()

def combine_gui(video_files, audio_files, config_path):
  output_textbox = sg.Multiline(size=(80,30), key='-OUTPUT-')
  layout = [[output_textbox],
            [sg.Button('Close', pad=(360,5))]]
  combine_window = sg.Window('Combining - describealign', layout, font=('Arial', 16),
                             disable_close=True, finalize=True)
  output_textbox.update('Combining media files:', append=True)
  print_queue = multiprocessing.Queue()
  
  settings = read_config_file(config_path)
  settings.update({'yes':True})
  proc = multiprocessing.Process(target=combine_print_exceptions,
                                 args=(print_queue, video_files, audio_files),
                                 kwargs=settings, daemon=True)
  proc.start()
  while True:
    # if the script isn't running anymore, re-enable the default close window button
    if not proc.is_alive():
      combine_window.DisableClose = False
    if not print_queue.empty():
      if IS_RUNNING_WINDOWS:
        cursor_position = output_textbox.WxTextCtrl.GetInsertionPoint()
      output_textbox.update(print_queue.get(), append=True)
      if IS_RUNNING_WINDOWS:
        output_textbox.WxTextCtrl.SetInsertionPoint(cursor_position)
    event, values = combine_window.read(timeout=100)
    # window closed event isn't always emitted, so also manually check window status
    if event == sg.WIN_CLOSED or combine_window.TKrootDestroyed:
      if proc.is_alive():
        proc.terminate()
      break
    if event == 'Close':
      if not proc.is_alive():
        combine_window.DisableClose = False
        break
      selection = sg.PopupYesNo('Combiner is still running, stop it and close anyway?')
      if selection != 'Yes':
        continue
      proc.terminate()
      combine_window.DisableClose = False
      break
  combine_window.close()

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

def main_gui():
  config_path = platformdirs.user_config_path(appname='describealign', appauthor=False, ensure_exists=True) / 'config.ini'
  old_paths = [
    # Place in chronological order (oldest -> newest)
    Path(__file__).resolve().parent / 'config.ini',
    platformdirs.user_config_path(appname='describealign', ensure_exists=True) / 'config.ini',
  ]
  
  # Get newest existent path
  old_config = next(
    (
      file 
      for file in reversed(old_paths)
      if file.exists()
    ), 
    None,
  )
  
  try:
    migrate_config(old_config, config_path)
  except OSError as exc:
    print(f"Error migrating old config:", *traceback.format_exception_only(exc))
    print(f"Old config left in place at {old_config}")
  
  sg.theme('Light Blue 2')
  
  filetype_sep = ';' if IS_RUNNING_WINDOWS else ' '
  all_audio_file_types = [('All Audio File Types', '*.' + f'{filetype_sep}*.'.join(AUDIO_EXTENSIONS)),]
  all_video_file_types = [('All Video File Types', '*.' + f'{filetype_sep}*.'.join(VIDEO_EXTENSIONS)),]
  all_video_and_audio_file_types = [('All Video and Audio File Types',
                                     '*.' + f'{filetype_sep}*.'.join(VIDEO_EXTENSIONS | AUDIO_EXTENSIONS)),]
  audio_file_types = [(ext, f"*.{ext}") for ext in AUDIO_EXTENSIONS]
  video_and_audio_file_types = [(ext, f"*.{ext}") for ext in VIDEO_EXTENSIONS] + audio_file_types
  audio_file_types = all_audio_file_types + audio_file_types
  video_and_audio_file_types = all_video_file_types + all_video_and_audio_file_types + video_and_audio_file_types
  # work around bug in PySimpleGUIWx's convert_tkinter_filetypes_to_wx function
  if IS_RUNNING_WINDOWS:
    file_fix = lambda file_types: file_types[:1] + [(f'|{type[0]}', type[1]) for type in file_types[1:]]
    audio_file_types = file_fix(audio_file_types)
    video_and_audio_file_types = file_fix(video_and_audio_file_types)
  
  layout = [[sg.Text('Select media files to combine:', size=(40, 2), font=('Arial', 20), pad=(3,15))],
            [sg.Column([[sg.Text('Video Input:', size=(11, 2), pad=(1,5)),
                         sg.Input(size=(35, 1.2), pad=(10,5), key='-VIDEO_FILES-',
                                  tooltip='List video filenames here, in order, separated by semicolons'),
                         sg.FilesBrowse(button_text="Browse Video",
                                        file_types=video_and_audio_file_types,
                                        tooltip='Select one or more video files')]], pad=(2,7))],
            [sg.Column([[sg.Text('Audio Input:', size=(11, 2), pad=(1,5)),
                         sg.Input(size=(35, 1.2), pad=(10,5), key='-AUDIO_FILES-',
                                  tooltip='List audio filenames here, in order, separated by semicolons'),
                         sg.FilesBrowse(button_text="Browse Audio",
                                        file_types=audio_file_types,
                                        tooltip='Select one or more audio files')]], pad=(2,7))],
            [sg.Column([[sg.Submit('Combine', pad=(40,3), tooltip='Combine selected video and audio files'),
                         sg.Button('Settings', tooltip='Edit settings for the GUI and algorithm.')]],
                         pad=((135,3),10))]]
  window = sg.Window('describealign', layout, font=('Arial', 16), resizable=False, finalize=True)
  window['-VIDEO_FILES-'].set_focus()
  while True:
    event, values = window.read()
    if event == 'Combine':
      if len(values['-VIDEO_FILES-']) == 0 or \
         len(values['-AUDIO_FILES-']) == 0:
        window.disable()
        sg.Popup('Error: empty input field.', font=('Arial', 20))
        window.enable()
        window['-VIDEO_FILES-'].set_focus()
        continue
      video_files = values['-VIDEO_FILES-'].split(';')
      if len(video_files) == 1:
        video_files = video_files[0]
      audio_files = values['-AUDIO_FILES-'].split(';')
      if len(audio_files) == 1:
        audio_files = audio_files[0]
      window.disable()
      combine_gui(video_files, audio_files, config_path)
      window.enable()
      window['-VIDEO_FILES-'].set_focus()
    if event == 'Settings':
      window.disable()
      settings_gui(config_path)
      window.enable()
      window['-VIDEO_FILES-'].set_focus()
    if event == sg.WIN_CLOSED:
      break
  window.close()

# Entry point for command line interaction, for example:
# > describealign video.mp4 audio_desc.mp3
def command_line_interface():
  if len(sys.argv) < 2
    if sg is not None:
      # No args, run gui
      print('No input arguments detected, starting GUI...')
      main_gui()
      sys.exit(0)
    else:
      print("Can't launch GUI and arguments missing.\nGUI dependencies missing.")
  
  parser = argparse.ArgumentParser(
                          description="Replaces a video's sound with an audio description.",
                          usage="describealign video_file.mp4 audio_file.mp3")
  parser.add_argument("video", help='A video file or directory containing video files.', nargs='?', default=None)
  parser.add_argument("audio", help='An audio file or directory containing audio files.', nargs='?', default=None)
  parser.add_argument('--smoothness', type=float, default=50,
                      help='Lower values make the alignment more accurate when there are skips ' + \
                           '(e.g. describer pauses), but also make it more likely to misalign. ' + \
                           'Default is 50.')
  parser.add_argument('--stretch_audio', action='store_true',
                      help='Stretches the input audio to fit the input video. ' + \
                           'Default is to stretch the video to fit the audio.')
  parser.add_argument('--keep_non_ad', action='store_true',
                      help='Tries to only replace segments with audio description. Useful if ' + \
                           'video\'s audio quality is better. Default is to replace all aligned audio. ' + \
                           'Requires --stretch_audio to be set, otherwise does nothing.')
  parser.add_argument('--boost', type=float, default=0,
                      help='Boost (or quieten) description volume. Units are decibels (dB), so ' + \
                           '-3 makes the describer about 2x quieter, while 3 makes them 2x louder. ' + \
                           'Requires --stretch_audio to be set, otherwise does nothing.')
  parser.add_argument('--ad_detect_sensitivity', type=float, default=.6,
                      help='Audio description detection sensitivity ratio. Higher values make ' + \
                           '--keep_non_ad more likely to replace aligned audio. Default is 0.6')
  parser.add_argument('--boost_sensitivity', type=float, default=.4,
                      help='Higher values make --boost less likely to miss a description, but ' + \
                           'also make it more likely to boost non-description audio. Default is 0.4')
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
  parser.add_argument("--extension", default="copy",
                      help='File type of output video (e.g. mkv). When set to "copy", copies the ' + \
                           'file type of the corresponding input video. Default is "copy".')
  parser.add_argument("--install-ffmpeg", action="store_true",
                      help="Install the required ffmpeg binaries and then exit. This is meant to be" + \
                           "run from a privileged installer process (e.g. OS X Installer)")
  args = parser.parse_args()
  
  if args.install_ffmpeg:
    # Make sure the file is world executable
    os.chmod(get_ffmpeg(), 0o755)
    os.chmod(get_ffprobe(), 0o755)
  elif args.video and args.audio:
    combine(args.video, args.audio, args.smoothness, args.stretch_audio, args.keep_non_ad,
            args.boost, args.ad_detect_sensitivity, args.boost_sensitivity, args.yes,
            args.prepend, args.no_pitch_correction, args.output_dir, args.alignment_dir,
            args.extension)
  else:
    parser.print_usage()

# allows the script to be run on its own, rather than through the package, for example:
# python3 describealign.py video.mp4 audio_desc.mp3
if __name__ == "__main__":
  multiprocessing.freeze_support()
  command_line_interface()




