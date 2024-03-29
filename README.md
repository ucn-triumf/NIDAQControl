# NIDAQ Control

<img src="https://img.shields.io/github/languages/top/ucn-triumf/NIDAQControl?style=flat-square"/> <img src="https://img.shields.io/github/languages/code-size/ucn-triumf/NIDAQControl?style=flat-square"/> <img alt="GitHub License" src="https://img.shields.io/github/license/ucn-triumf/NIDAQControl"> <img src="https://img.shields.io/github/last-commit/ucn-triumf/NIDAQControl?style=flat-square"/>


Python API for NIDAQ devices used by the TUCAN collaboration.

## Installation

You need to first install the NI-DAQmx drivers from [their webpage](https://www.ni.com/en/support/downloads/drivers/download.ni-daq-mx.html#521556).

Clone this directory then execute the following:

```
cd NIDAQControl
pip install .
```

Import as

```
import NIDAQControl
```

## Documentation

See [documentation index here](docs/NIDAQControl/index.md)

## Examples

### Read data

When no analog output is specified, default to output 0 V on `ao0`.

```python
from NIDAQControl import USB6281

# make object
u = USB6281()

# setup connections
# pattern: ch : label
# where ch is the channel number (i.e. 4 = ai4) and
# label is the human-readable item the channel is connected to
u.setup(ai = {  1: "Input a",
                4: "Input b",
             })

# run
u.run(  duration = 10, # run for this many seconds
        draw_s = 2, # draw the last two seconds
        sample_freq = 1000, # while the NIDAQ by default measures at 20kHz, we downsample to 1000 Hz in software.
        draw_ch_top = 'Input a' # force input a to be drawn on top and more easily visible
     )

u.to_csv() # write to file with default filename
```

### Write data

When no analog input is specified, default to reading `ai0`. Decrease sample rate for long output runs.

```python
from NIDAQControl import USB6281

# make object
u = USB6281()

# setup connections
# pattern: ch : function
# where ch is the channel number (i.e. 0 = ao0) and
# function is a python function which takes as only input time
u.setup(ao = {  0: lambda time : 3,  # output a constant 3V
                1: lambda time : time % 2  # sawtooth from 0 to 1 with a period of 2 seconds
              })

# run
u.run(duration = 10) # run for this many seconds
```

### Read and write data with signal filtering

```python
from NIDAQControl import USB6281

# make object
u = USB6281()

# setup connections
u.setup(ao = {0: lambda time : np.sin(2*np.pi*10*time)},  # output a 10 Hz sine wave
        ai = {0: "Output from ao0"})

# setup various filters (they are all applied in sequence)
u.set_filter(low = 2000)  # low pass
u.set_filter(high = 1)  # high pass
u.set_filter(low = 50, high = 1000)  # band pass
u.set_filter(low = 100, high = 120, bandstop=True)  # band stop

# run
u.run(  duration = 5, # run for this many seconds
        draw_s = 5,
        sample_freq = 100, # This parameter used only for read. Write always outputs at the clock frequency
     )

# write to file with specified filename and some notes
# note keywords are arbitrary
u.to_csv('test.csv')
```

