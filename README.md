# describealign
Combines videos with matching audio files (e.g. audio descriptions).  Works by aligning parts of the audio file to matching parts of the video's sound.

## Quickstart

Create a copy of a video file with the sound replaced by an audio description:
```
describealign video.mp4 audio_desc.mp3
```

## Installation

### package method

This script is compatible with Python versions 3.8 and up. Versions before that won't work (it relies on a recent update to Scipy's linprog).

describealign is available with pip:
```bash
pip install describealign
```
Note: You may need to add the folder Python's pip.exe is in to your system path. It might be something like: "C:/Users/<User>/AppData/Local/Programs/Python/Python310/Scripts"

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

## Testing Installation

The installation can be tested on the 1929 comedy short [Ask Dad](https://archive.org/details/ask_dad), with [audio description](https://archive.org/details/MoviesForTheBlind01-askDad) provided by Valerie H. in her podcast [Movies For the Blind:](https://moviesfortheblind.com/)

```bash
describealign test_media/ask_dad.mp4 test_media/ask_dad_moviesfortheblind_ep_01.mp3
```

This produces two outputs, a new video file "videos_with_ad/ad_ask_dad.mp4" and a plot in alignment_plots:

<img src="https://raw.githubusercontent.com/kkroening/describealign/master/readme_media/ask_dad.png" alt="Ask Dad Alignment" align="middle" width="50%"/>

## Advanced Usage

### boost

describealign also has a few other experimental capabilities, like boosting the volume of audio descriptions relative to the video's sound with the "--boost x" argument, where x is in decibels. "--boost 3" approximately doubles the audio description volume, while "--boost -3" approximately halves it.

### keep_non_ad

The default behavior of describealign is to replace all or almost all of a video's audio with the audio description file's audio. But the "--keep_non_ad" argument tells describealign to try to only replace audio when the describer is speaking. This can be useful when the audio description has significantly worse sound quality than the video.

### additional arguments

If an alignment isn't working perfectly, the ambitious user can try adjusting a few parameters with arguments described in "--help".
