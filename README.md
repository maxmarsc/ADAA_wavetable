# Experimentations on ADAA for wavetable oscillators
This repository contains all the python experiments I made based on the IEEE research paper [Antiderivative Antialiasing for Arbitrary Waveform Generation](https://ieeexplore.ieee.org/document/9854137)

The paper provided an algorithm, some results and some matlab demo code which you can find [here](https://dangelo.audio/ieee-talsp-aaiir-osc.html)

This work was presented at the [Audio Developer Conference 2023](https://audio.dev/conference/) alongside its C++ implementation for real-time [here](https://github.com/maxmarsc/libadawata). 

The code contained in here is certainly not production ready, but I made what I could to understand, replicate, and further adapt the algorithm to a real-time scenario.

*Keep in mind that I'm not a DSP specialist, if you find something weird or buggy in my code don't hesitate to tell it. Also this repository is not dedicated to explain the algorithm in any case.*

## What's included
This repository contains 3 mains parts :
- `matlab` : contains the Matlab code of the paper demo, slightly modified to run with Octave
- `python` : contains the different versions of the algorithm and some tools to
analyze the results (metrics, graphs...)
- `python/legacy.py` : contains some iterations of my work when adapting the algorithm.
It's only provided for R&D legacy and should not be considered reliable

## Python experimentations
### Requirements
The following tools are required :
- `libsamplerate` : for mipmapping resampling
- `libsndfile` : for audio exporting

On Ubuntu you can install `libsamplerate` and `libsndfile` with the following command:
```bash
apt-get install -y libsamplerate0 libsndfile1
```

After that you will need to install the python requirements :
```bash
pip install -r requirements.txt
```

### How to use
I provide a main python script that can performs three tasks, on different version
of both the ADAA algorithm, and its alternatives (lerp + oversampling) :
- Metrics computation (SNR and SINAD)
- Sweep test spectrogram plot
- Power spectral density plot

Some values still needs to be modified manually in the `main.py` file depending on your use case:
- `DURATION_S` : The duration of generated audio, might lead to high ram usage if too high
- `FREQS` : A list of frequencies to generate for (only in psd/metrics modes)
- `ALGOS_OPTIONS` :  A list of all the algorithm to test
- `NUM_PROCESS` : The number of parallel process, maxed out to 20, mined out to you ncpus
- `SAMPLERATE`

#### Metrics
For the metrics mode use the following options :
```bash
python python/main.py metrics [--export {snr,sinad,both,none}] [--export-dir EXPORT_DIR] [--export-audio] [--export-phase]
```

You'd usually want to add all the frequencies you want to test in `FREQS`.
The script will write the metrics in CSV files.

#### Sweep test
For the metrics mode use the following options :
```bash
python python/main.py sweep [--export-dir EXPORT_DIR] [--export-audio] [--export-phase]
```

This will automatically generate a sweep test from 20Hz to Nyquist and plot its spectrogram.  
This mode will not read the `FREQS` variable.  
I suggest a duration of 5s to have a good enough resolution in the spectrogram.

#### PSD
For the psd mode use the following options :
```bash
python python/main.py psd [--export-dir EXPORT_DIR] [--export-audio] [--export-phase]
```

This will use a matplotlib graph to display the psd values for each test, and a final
graph with all the waveforms on the same graphs.

**This mode requires `FREQS` to contains a single value**

# What's next
As mentioned above, this is an experimentation repo, not a tool designed for
advanced use or anything like it.

I don't plan to make modifications to make it a user-friendly demo tool.
However I'm open to suggestions in order to help further researchs such as :
- Improvements on the argparser to allow passing frequencies and/or other parameters
- ~~Metrics improvements/fixes~~ DONE
- Improvement/Changes in the algorithm


If you wan't to discuss about it you can open an issue or you can find me on :
 - [Discord](https://discordapp.com/users/Groumpf#2353)
 - [Twitter](https://twitter.com/Groumpf_)
 - [Mastodon](https://piaille.fr/@groumpf)
