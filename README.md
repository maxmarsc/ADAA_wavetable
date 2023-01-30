# Purpose
This repository contains all the experiments I made based on the IEEE research paper [Antiderivative Antialiasing for Arbitrary Waveform Generation](https://ieeexplore.ieee.org/document/9854137)

The paper provided an algorithm, some results and some matlab demo code which you can find [here](https://dangelo.audio/ieee-talsp-aaiir-osc.html)

The code contained in here is certainly not production ready, but I made what I could to understand and try to replicate the paper code in the most comprehensible and real-time compatible way.

TLDR; check `python/main.py::process_fwd` for a simpler python implementation.

*Keep in mind that I'm not a DSP specialist of any kind, if you find something weird or buggy in my code don't hesitate to tell it. Also this repository is not dedicated to explain the algorithm in any case.*


## The matlab folder
I don't own a Matlab license, so I used Octave to run the demo code. I had to make a few modifications in the file in order for Octave to be able to parse it.

Also I fixed a few typos in the demo code.

I added a `generateWavetableSaw()` method to replicate a naive saw wave from a wavetable, to process it like a real "arbitrary" wavetable.

To run the demo with octave, you will need the `signal` package. Then run 
```shell
octave AAIIR_demo.m
```

## The python folder
Most of my experiment were made in Python. I recoded the algorithm from matlab to python. Then I made a second version a bit simpler, focusing on making suitable for real time implementation (the base algorithm have some ever-increasing indexes which is kinda hard to support).

Check `python/main.py` for the details, I tried to document it so it's self-explanatory.

You can find the python requirements in the `requirements.txt` file. You might also need `libsoxr`.

# Results
I was able to get good results with butterworth filtering, but not with Chebyshev type-2 filtering. Whereas in the paper they claim 10th order Chebyshev had incredible results I was not able to reproduce it. *This might be because I'm missing something*


For now I won't go any further because the computationnal cost would be too big for my embedded project and the results aren't good enough so far.

If you wan't to discuess about it you can find me on :
 - [Discord](https://discordapp.com/users/Groumpf#2353)
 - [Twitter](https://twitter.com/Groumpf_)
 - [Mastodon](https://piaille.fr/@groumpf)
