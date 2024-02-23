# NIDAQ Control

Python API for NIDAQ devices used by the TUCAN collaboration.

## Installation

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

Read data

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
u.close() # close connections to the NIDAQ nicely
```

Write data

```python
from NIDAQControl import USB6281

# make object
u = USB6281()

# setup connections
# pattern: ch : function
# where ch is the channel number (i.e. 0 = ao0) and
# function is a python function which takes as only input time
u.setup(ao = {  0: lambda time : 3  # output a constant 3V
                1: lambda time : time % 2  # sawtooth from 0 to 1 with a period of 2 seconds
             })

# run
u.run(  duration = 5, # run for this many seconds
     )

u.close() # close connections to the NIDAQ nicely
```

Read and write data

```python
from NIDAQControl import USB6281

# make object
u = USB6281()

# setup connections
u.setup(ao = {  0: lambda time : np.sin(2*np.pi*10*time)  # output a 10 Hz sine wave
             }
        ai = {0: "Output from ao0"})

# run
u.run(  duration = 5, # run for this many seconds
        draw_s = 5,
        sample_freq = 100, # This parameter used only for read. Write always outputs at the clock frequency
     )

# write to file with specified filename and some notes
# note keywords are arbitrary
u.to_csv('test.csv',
         note1 = "Testing",
         connections = "ai0 connected to ao0",
         fdskjlksdfnsd = "Use your own keywords")
u.close() # close connections to the NIDAQ nicely
```