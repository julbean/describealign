# describealign
Combines videos with matching audio files (e.g. audio descriptions). Works by aligning parts of the audio file to matching parts of the video's sound.


## Quickstart

Create a copy of a video file with the sound replaced by an audio description:

<img src="https://github.com/julbean/describealign/blob/main/readme_media/describealign_gui_main.PNG" alt="GUI main" align="middle" width="50%"/>

Select a video file and a corresponding audio description using the file browsers, then click "Combine":

<img src="https://github.com/julbean/describealign/blob/main/readme_media/describealign_gui_combiner.PNG" alt="GUI combiner" align="middle" width="50%"/>

The combined media is saved in the folder "videos_with_ad" placed in the directory describealign was run in. The directory that combined media files are saved in can be changed in "Settings":

<img src="https://github.com/julbean/describealign/blob/main/readme_media/describealign_gui_settings.PNG" alt="GUI combiner" align="middle" width="50%"/>

Note: media longer than an hour should only be processed on computers with at least 16 GB of RAM.


## Installation

### package method

This script is compatible with Python versions 3.8 and up. Versions before that won't work (it relies on a recent update to Scipy's linprog).

describealign is available with pip:
```bash
pip install describealign
```
Note: You may need to add the folder Python's pip.exe is in to your system path. It might be something like: "C:/Users/User/AppData/Local/Programs/Python/Python310/Scripts" Don't forget to restart command prompt after updating the PATH!

The GUI can then be opened from console/command prompt in any directory with:
```bash
describealign
```
Note: You may need to add the folder Python's compiled scripts (e.g. describealign.exe) are kept in to your system path. It may be a different folder than pip's, like AppData/Roaming instead of AppData/Local.

### script method

Alternatively, the python script (describealign.py) can be downloaded from here and run directly after installing the dependencies manually (requirements.txt):
```bash
pip install -r requirements.txt
python3 describealign.py
```

### binary method

The binary methods don't require installing python or messing about with PATH.

Windows and Mac users can instead download and unzip the [latest release](https://github.com/julbean/describealign/releases/latest), then double click on describealign.exe to open the GUI.

Note for Mac binary users: To open the binary, you'll need to ctrl+click (or right click) on the binary, then click "Open" and then click "Open" again in the window that pops up.  This minor annoyance is a result of my unwillingness to pay Apple $100 a year.  That also causes three other quirks with the Mac binary: a console window will pop up first and the GUI will only appear after about a minute, it saves outputs in the home folder (i.e. the one with your username) by default, so to change that you'll need to choose an output folder in describealign settings, and it also has to redownload ffmpeg each time it's opened.

### updating

When new versions are released, the pip package can be updated with:
```bash
pip install describealign --upgrade
```

Note: users with multiple python versions may need to use pip3 rather than pip.


## Testing Installation

The installation can be tested on a clip from the 1929 comedy short [Ask Dad](https://archive.org/details/ask_dad), with the first part of an [audio description](https://archive.org/details/MoviesForTheBlind01-askDad) provided by Valerie H. in her podcast [Movies For the Blind.](https://moviesfortheblind.com/) Download the trimmed versions from the test_media folder in this repository, then select them in the GUI:

<img src="https://github.com/julbean/describealign/blob/main/readme_media/describealign_gui_main_filled.PNG" alt="GUI main filled" align="middle" width="50%"/>

This produces two outputs, a new video file "videos_with_ad/ad_ask_dad_trimmed.mp4" and a plot in alignment_plots:

<img src="https://github.com/julbean/describealign/blob/main/readme_media/ask_dad_trimmed.png" alt="Ask Dad Trimmed Alignment" align="middle" width="50%"/>

The plot shows the audio description was already aligned with the video apart from a fixed offset of 199 seconds, which means Valerie starts describing Ask Dad 199 seconds into the episode. The y-scale is so zoomed in that the sub-second dithering of tokens (used to fine-tune alignment) is visible as blue streaks.

If the full video (22 minutes) and audio description (27 minutes) are used instead, describealign runs in about 90 seconds, using up about 3 GB of RAM, and we get the following plot:

<img src="https://github.com/julbean/describealign/blob/main/readme_media/ask_dad.png" alt="Ask Dad Alignment" align="middle" width="50%"/>

This plot shows a number of small pauses in the audio description starting around 10 minutes in, which add up to a total offset of 30 seconds by the end of the video. The jump discontinuities have been smoothed out by stretching the video. The plot also shows which segments of audio would be replaced if --stretch_audio were used. All of the audio would be replaced except for a segment around the 9 minute mark in which the video's original audio would be kept, as the replacement audio would have been too noticably distorted (i.e. more than 10% stretched).

A text version of each plot is saved alongside each image:

```
Main changes needed to video to align it to audio input:
Start Offset: 199.06 seconds
Median Rate Change: 0.00%
Rate change of    0.0% from  0:00:00.00 to  0:09:08.64 aligning with audio from  0:03:19.06 to  0:12:27.67
Rate change of   11.0% from  0:09:08.64 to  0:10:07.34 aligning with audio from  0:12:27.67 to  0:13:20.56
Rate change of    0.4% from  0:10:07.34 to  0:10:51.53 aligning with audio from  0:13:20.56 to  0:14:04.60
Rate change of    4.7% from  0:10:51.53 to  0:13:48.45 aligning with audio from  0:14:04.60 to  0:16:53.65
Rate change of    0.2% from  0:13:48.45 to  0:14:48.80 aligning with audio from  0:16:53.65 to  0:17:53.89
Rate change of    6.4% from  0:14:48.80 to  0:17:42.84 aligning with audio from  0:17:53.89 to  0:20:37.44
Rate change of    0.4% from  0:17:42.84 to  0:18:52.56 aligning with audio from  0:20:37.44 to  0:21:46.88
Rate change of    4.3% from  0:18:52.56 to  0:20:02.88 aligning with audio from  0:21:46.88 to  0:22:54.31
Rate change of   -0.0% from  0:20:02.88 to  0:20:52.45 aligning with audio from  0:22:54.31 to  0:23:43.91
Rate change of   -2.9% from  0:20:52.45 to  0:21:38.09 aligning with audio from  0:23:43.91 to  0:24:30.92
Rate change of   -0.1% from  0:21:38.09 to  0:22:15.77 aligning with audio from  0:24:30.92 to  0:25:08.63
```


## Advanced Usage

### directories

describealign can be given a directory of videos and a directory of audio files rather than individual files. describealign assumes files from the two directories correspond based on their lexicographic order.

### stretch_audio (audio-to-video alignment)

By default describealign stretches video to fit audio descriptions, but the inverse is also possible: stretching the audio description to fit the video with the "--stretch_audio" argument.

### audio-to-audio

Whereas describealign is designed to align video-to-audio, it can also align an audio file to another audio file.

### boost

When using the --stretch_audio argument, describealign also has a few other experimental capabilities, like boosting the volume of audio descriptions relative to the video's sound with the "--boost x" argument, where x is in decibels. "--boost 3" approximately doubles the audio description volume, while "--boost -3" approximately halves it.

### keep_non_ad

The default behavior of --stretch_audio is to replace all or almost all of a video's audio with the audio description file's audio. But the "--keep_non_ad" argument tells describealign to try to only replace audio when the describer is speaking. This can be useful when the audio description has significantly worse sound quality than the video.

### additional arguments

If an alignment isn't working perfectly, the ambitious user can try adjusting a few parameters with arguments described in "--help" and the GUI's Settings tooltips.

### command line interface

describalign can be run without the GUI by specifying input media as positional arguments:
```bash
describealign video.mp4 audio_desc.mp3
```

### module

describealign can also be used as a python module:
```python
import describealign as dal
dal.combine("ask_dad_trimmed.mp4", "ask_dad_moviesfortheblind_ep_01_trimmed.mp3")
```


## Interesting Use Cases

### dub alignment

describealign is robust enough to align media with completely different dialogue, meaning it can align audio dubbed in a different language to the original video.

### lossless video editing

With default settings (i.e. --stretch_audio set to False), describealign doesn't re-encode either the video or audio streams. It aligns them by modifying the timestamps that video frames are shown at, which means no loss in quality. Basic video editing can be done by deleting or stretching segments of a video's sound in Audacity, then running describealign on the original video and the modified audio with --smoothness set low (e.g. 1). The video will be edited losslessly, but the audio can also be edited losslessly by exporting from Audacity as FLAC, then setting --extension to mkv or another container that supports FLAC.

### isolating descriptions for transcription

By using a very high boost value (e.g. --boost 100), the output audio will only contain the audio descriptions and all other sounds will be silenced. Passing the output into [Whisper](https://github.com/openai/whisper) will then create a transcript of just the audio descriptions.



