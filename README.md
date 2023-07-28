# describealign
Combines videos with matching audio files (e.g. audio descriptions). Works by aligning parts of the audio file to matching parts of the video's sound.


## Quickstart

Create a copy of a video file with the sound replaced by an audio description:
```
describealign video.mp4 audio_desc.mp3
```

Note: media longer than an hour should only be processed on computers with at least 16 GB of RAM.


## Installation

### package method

This script is compatible with Python versions 3.8 and up. Versions before that won't work (it relies on a recent update to Scipy's linprog).

describealign is available with pip:
```bash
pip install describealign
```
Note: You may need to add the folder Python's pip.exe is in to your system path. It might be something like: "C:/Users/User/AppData/Local/Programs/Python/Python310/Scripts" Don't forget to restart command prompt after updating the PATH!

The script can then be run from console/command prompt in any directory with:
```bash
describealign
```
Note: You may need to add the folder Python's compiled scripts (e.g. describealign.exe) are kept in to your system path. It may be a different folder than pip's, like AppData/Roaming instead of AppData/Local.

### script method

Alternatively, the python script (describealign.py) can be downloaded from here and run directly after installing the dependencies manually (requirements.txt):
```bash
pip install -r requirements.txt
python3 describealign.py video.mp4 audio_desc.mp3
```

### updating

When new versions are released, the pip package can be updated with:
```bash
pip install describealign --upgrade
```

Note: users with multiple python versions may need to use pip3 rather than pip.


## Testing Installation

The installation can be tested on a clip from the 1929 comedy short [Ask Dad](https://archive.org/details/ask_dad), with the first part of an [audio description](https://archive.org/details/MoviesForTheBlind01-askDad) provided by Valerie H. in her podcast [Movies For the Blind.](https://moviesfortheblind.com/) Download the trimmed versions from the test_media folder in this repository, change to the directory with the files and run:

```bash
describealign ask_dad_trimmed.mp4 ask_dad_moviesfortheblind_ep_01_trimmed.mp3
```

This produces two outputs, a new video file "videos_with_ad/ad_ask_dad_trimmed.mp4" and a plot in alignment_plots:

<img src="https://github.com/julbean/describealign/blob/main/readme_media/ask_dad_trimmed.png" alt="Ask Dad Trimmed Alignment" align="middle" width="50%"/>

The plot shows the audio description was already aligned with the video apart from a fixed offset of 199 seconds, which means Valerie starts describing Ask Dad 199 seconds into the episode. The y-scale is so zoomed in that the sub-second dithering of tokens (used to fine-tune alignment) is visible as blue streaks.

If the full video (22 minutes) and audio description (27 minutes) are used instead, describealign runs in about 90 seconds, using up about 3 GB of RAM, and we get the following plot:

<img src="https://github.com/julbean/describealign/blob/main/readme_media/ask_dad.png" alt="Ask Dad Alignment" align="middle" width="50%"/>

This plot shows a number of small pauses in the audio description starting around 10 minutes in, which add up to a total offset of 30 seconds by the end of the video. The jump discontinuities have been smoothed out by stretching the audio description. All of the audio was replaced except for a segment around the 9 minute mark in which the video's original audio was kept, as the replacement audio would have been too noticably distorted (i.e. more than 10% stretched).


## Advanced Usage

### directories

describealign can be given a directory of videos and a directory of audio files rather than individual files. describealign assumes files from the two directories correspond based on their lexicographic order.

### boost

describealign also has a few other experimental capabilities, like boosting the volume of audio descriptions relative to the video's sound with the "--boost x" argument, where x is in decibels. "--boost 3" approximately doubles the audio description volume, while "--boost -3" approximately halves it.

### keep_non_ad

The default behavior of describealign is to replace all or almost all of a video's audio with the audio description file's audio. But the "--keep_non_ad" argument tells describealign to try to only replace audio when the describer is speaking. This can be useful when the audio description has significantly worse sound quality than the video.

### additional arguments

If an alignment isn't working perfectly, the ambitious user can try adjusting a few parameters with arguments described in "--help".

### module

describealign can also be used as a python module:

```python
import describealign as dal
dal.combine("ask_dad_trimmed.mp4", "ask_dad_moviesfortheblind_ep_01_trimmed.mp3")
```

## Planned Features

### Video-to-Audio Alignment

Currently, describealign stretches audio descriptions to fit video, but the inverse should also be possible: stretching video to fit audio description. A future version will include this feature.

### GUI

Many describealign users are describers or family members of the visually impaired, so an optional, cross-platform Graphical User Interface could improve usability.






