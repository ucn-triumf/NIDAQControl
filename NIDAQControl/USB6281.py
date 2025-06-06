# Read and write from NIDAQ Ni USB 6281
# Specification sheet: https://www.ni.com/docs/en-US/bundle/pci-pxi-usb-6281-specs/page/specs.html#GUID-DAEADA0E-7005-4D59-BB9E-8C9F3438874E__GUID-99EE1305-926C-41F0-97E0-570060200F05
#
# Derek Fujimoto
# Feb 2024

import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nidaqmx as ni
from tqdm import tqdm, TqdmWarning
from datetime import datetime
from scipy.signal import butter, sosfiltfilt
import os, time, __main__

from nidaqmx import stream_readers
from nidaqmx import stream_writers

import warnings
warnings.simplefilter('ignore', category=TqdmWarning)

# set up graphs
plt.ion()
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = True
mpl.rcParams['axes.spines.top'] = True
mpl.rcParams['axes.spines.bottom'] = True

class USB6281(object):
    """Read and set values on analog channels of NI USB 6281 DAQ.

    Public Attributes:
        ai (dict): saved inputs from setup
        ao (dict): saved inputs from setup
        df (pd.DataFrame): data saved from ai channels
        signal_filters (list): list of second-order section filters corresponding to a butterworth filter. Applied in order.
        signal_generator (dict): dictionary of generators to set the voltages of the NIDAQ
    """

    # arguments to pass to add_ao_voltage_chan
    _ao_args = {'min_val': -10,
                'max_val': 10}

    _ai_args = {'min_val': -10,
                'max_val': 10}

    # set the number of frames in a buffer (override)
    # the data gets written in chunks, each chunk is a frame
    # NOTE  With my NI6211 it was necessary to override the default buffer
    # size to prevent under/over run at high sample rates
    _frames_per_buffer = 10

    def __init__(self, device_name = 'Dev1',
                       clock_freq = 2e4,
                       samples_per_channel = 1e3,
                       terminal_config = 'RSE'):
        """Initialize object

        Args:
            device_name (str): name of device which to connect
            clock_freq (int): rate of data taking and output in Hz. Needs to be sufficiently high to
                        prevent write errors 1000 is too low, 100000 is good.
            samples_per_channel (int): set buffer size. If you specify samples per channel of 1000 samples
                        and your application uses two channels, the buffer size would be 2000 samples.
                        See https://documentation.help/NI-DAQmx-Key-Concepts/bufferSize.html
                        Requres that samples_per_channel > clock_freq // samples_per_channel.
            terminal_config (str): string, one of DEFAULT, DIFF, NRSE, PSEUDO_DIFF, RSE. According to
                        documentation: https://nidaqmx-python.readthedocs.io/en/latest/constants.html#nidaqmx.constants.CalibrationTerminalConfig,
                        DEFAULT appears to be DIFF. Typically we want RSE (common ground).

        Returns:
            None
        """

        # save inputs
        self._device_name = device_name
        self._clock_freq = int(clock_freq)
        self._samples_per_channel = int(samples_per_channel)
        self._terminal_config = getattr(ni.constants.TerminalConfiguration,
                                        terminal_config)

        # samples per frame
        self._samples_per_frame = self._clock_freq // self._frames_per_buffer

        # defaults
        self.ai = None
        self.ao = None
        self.signal_filters = []
        self.do_filter = False

        # timesteps from t=0 for output functions
        self._timebase_ao = np.arange(self._samples_per_frame, dtype=np.float64) / self._clock_freq

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def _filter(self, data):
        """Filter an array

        Args:
            data (array): array to filter
        Returns:
            array: with signal filters applied
        """

        # trivial end
        if not self.do_filter:
            return data

        # apply filters
        data = np.copy(data)
        for sos in self.signal_filters:
            data = sosfiltfilt(sos, data, padtype='even')

        return data

    def _draw_in_progress(self):
        """Update a figure while the run is ongoing

        Returns:
            bool: True if continue drawing, else close connection to nidaq and halt
        """

        # get obj
        fig = self._ax.figure

        # downsample only if filtering
        downsmpl = self._downsample if self.do_filter else 1

        # update data indexes
        i = self._nbuffers_read*self._len_buffer
        i_start = max(i-self._ydata.shape[1]*downsmpl, 0)

        # get data to copy, trim, filter
        data = self._filter(self._data[:, :i])
        try:
            self._ydata[:, (i_start-i)//downsmpl:] = data[:, i_start::downsmpl]

        # broadcast fails for data not even multiple of downsampled array
        except ValueError:
            n = len(data[0, i_start::downsmpl])
            self._ydata[:, -n:] = data[:, i_start::downsmpl]

        # set data
        for y, line in zip(self._ydata, self._ax.lines):
            line.set_ydata(y)

        # # set limits not too frequently
        ylim_low = np.min(self._ydata)
        ylim = self._ax.get_ylim()
        if not (ylim_low < ylim[0] or ylim_low > ylim[0] + abs(ylim[0]) * 0.1):
            ylim_low = ylim[0]
        else:
            if ylim_low > 0:
                ylim_low *= 0.9
            else:
                ylim_low *= 1.1

        ylim_high = np.max(self._ydata)
        if not (ylim_high > ylim[1] or ylim_high < ylim[1] - abs(ylim[1]) * 0.1):
            ylim_high = ylim[1]
        else:
            if ylim_high > 0:
                ylim_high *= 1.1
            else:
                ylim_high *= 0.9

        self._ax.set_ylim((ylim_low, ylim_high))

        # update canvas
        fig.canvas.draw()
        fig.canvas.flush_events()

        # check if window is still up, if not stop operation
        return plt.fignum_exists(self._ax.figure.number)

    def _make_signal_generator(self, fn_handle):
        """Makes a function which yields voltages for each of the analog output channels

        Args:
            fn_handle (function handle): function of the format voltage = fn(time) which specifies the output voltage

        Returns:
            iterator: function which yields voltage output values
        """
        # track phase in time domain
        phase = 0
        phase_step = self._samples_per_frame/self._clock_freq

        # generate voltages for ever and ever
        while True:

            # call the function
            output = fn_handle(self._timebase_ao+phase)

            # handle single value return types
            if type(output) in (int, float, np.float64):
                output = np.full(self._timebase_ao.shape, output, dtype=np.float64)
            yield output
            phase += phase_step

    def _read_task_callback(self, task_handle, every_n_samples_event_type,
                           number_of_samples, callback_data):
        """Read data callback. Set up for a register_every_n_samples_acquired_into_buffer_event event.

        Args:
            task_handle: handle to the task on which the event occurred.
            every_n_samples_event_type: EveryNSamplesEventType.ACQUIRED_INTO_BUFFER value.
            number_of_samples parameter:the value you passed in the sample_interval parameter
                                        of the register_every_n_samples_acquired_into_buffer_event.
            callback_data: apparently unused, but required

        Returns:
            int: specifically zero. Required by nidaqmx documentation
        """

        # read samples
        self._stream_in.read_many_sample(self._buffer_in,
                                        number_of_samples,
                                        timeout=ni.constants.WAIT_INFINITELY)

        # downsample buffer only if not filtering
        times = np.arange(self._samples_per_channel) + self._total_pts
        self._total_pts += self._samples_per_channel

        if self.do_filter:
            buffer_down = self._buffer_in

        else:
            times = times[::self._downsample]
            buffer_down = self._buffer_in[:, ::self._downsample]

        # save data
        i = self._nbuffers_read
        try:
            self._data[:, i*self._len_buffer:(i+1)*self._len_buffer] = buffer_down
            self._time[i*self._len_buffer:(i+1)*self._len_buffer] = times

        # at end of storage array
        except ValueError as err:
            if self._data[:, i*self._len_buffer:(i+1)*self._len_buffer].shape[1] == 0:
                pass
            else:
                raise err from None

        self._nbuffers_read += 1

        # Absolutely needed for this callback to be well defined (see nidaqmx doc).
        return 0

    def _setup(self):
        """Set up channels for input and output"""

        # must always read back at least one channel
        if self.ai is None:
            self.ai = {0: 'ai0'}

        # must always output one channel
        if self.ao is None:
            self.ao = {0: lambda x: 0}

        # setup output tasks -----------------------------------------------

        # number of channels
        self._len_ai = len(self.ai.keys())

        # task
        self._taski = ni.Task()

        # setup channels
        for ch in self.ai.keys():
            self._taski.ai_channels.add_ai_voltage_chan(f"{self._device_name}/ai{ch}",
                                                       **self._ai_args )

        # set terminal configuation
        self._taski.ai_channels.all.ai_term_cfg = self._terminal_config

        # setup clocks
        self._taski.timing.cfg_samp_clk_timing(rate = self._clock_freq,
                                     source = f'/{self._device_name}/ao/SampleClock',
                                     sample_mode = ni.constants.AcquisitionType.CONTINUOUS,
                                     samps_per_chan = self._samples_per_channel)

        # get stream
        self._stream_in = stream_readers.AnalogMultiChannelReader(self._taski.in_stream)

        # setup reading callback
        # read data when n samples are placed into the buffer
        self._taski.register_every_n_samples_acquired_into_buffer_event(self._samples_per_channel,
                                                                        self._read_task_callback)

        # setup output tasks -----------------------------------------------

        # number of channels
        self._len_ao = len(self.ao.keys())

        # task
        self._tasko = ni.Task()

        # setup channels
        for ch in self.ao.keys():
            self._tasko.ao_channels.add_ao_voltage_chan(f'{self._device_name}/ao{ch}',
                                                   **self._ao_args)

        # clock
        self._tasko.timing.cfg_samp_clk_timing(rate = self._clock_freq,
                                     sample_mode = ni.constants.AcquisitionType.CONTINUOUS,
                                     samps_per_chan = self._samples_per_channel)

        # stream
        self._stream_out = stream_writers.AnalogMultiChannelWriter(self._tasko.out_stream)

        # setup output buffer
        self._tasko.out_stream.output_buf_size = self._clock_freq
        self._tasko.out_stream.regen_mode = ni.constants.RegenerationMode.DONT_ALLOW_REGENERATION

        # setup output callback
        self._tasko.register_every_n_samples_transferred_from_buffer_event(self._samples_per_frame,
                                                                           self._write_task_callback)

    def _write_task_callback(self, task_handle, every_n_samples_event_type,
                           number_of_samples, callback_data):
        """Write data callback. Set up for a register_every_n_samples_transferred_from_buffer_event event.

        Args:
            task_handle: handle to the task on which the event occurred.
            every_n_samples_event_type: EveryNSamplesEventType.ACQUIRED_INTO_BUFFER value.
            number_of_samples parameter:the value you passed in the sample_interval parameter
                                        of the register_every_n_samples_acquired_into_buffer_event.
            callback_data: unused

        Returns:
            int: specifically zero. Required by nidaqmx documentation
        """

        # generate signal
        signal = np.array([next(self.signal_generator[ch]) for ch in self.ao.keys()],
                          dtype=np.float64)

        # save signal for debugging
        if self._save_ao:
            self._output_voltages.append(signal)

        # write signal to device
        self._stream_out.write_many_sample(signal,  timeout=1)

        # Absolutely needed for this callback to be well defined (see nidaqmx doc).
        return 0

    def close(self):
        """Close tasks"""
        try:
            self._taski.stop()
            self._taski.close()
        except ni.DaqError:
            pass

        try:
            self._tasko.stop()
            self._tasko.close()
        except ni.DaqError:
            pass

    def draw_data(self, cols=None, do_filter=True, do_downsample=True, **df_plot_kw):
        """Draw data in axis

        Args:
            cols (list): list of column names (str) to draw. If none, draw all
            do_filter (bool): if true, if there is a filter, then apply it before drawing
            do_downsample (bool): if true, apply downsampling before drawing (if no filtering, this option forced True)
            df_plot_kw: keywords to pass to pd.DataFrame.plot


        Returns:
            None, draws data to figure
        """

        # copy
        df = self.df.copy()

        # filter
        if self.do_filter and do_filter:
            for c in df.columns:
                df[c] = self._filter(df[c])

        # downsample
        if self.do_filter and do_downsample:
            df = df.loc[::self._downsample]

        if cols is not None:
            df[cols].plot(**df_plot_kw)
        else:
            df.plot(**df_plot_kw)

        # plot elements
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage Readback (V)')
        plt.legend(fontsize='x-small')
        plt.tight_layout()

    def draw_intended_output(self, **plot_kw):
        """Draw the values sent to ao, useful for debugging

        Args:
            plot_kw: keywords passed to plt.plot

        Returns:
            None
        """

        # check that data is saved
        if not self._save_ao:
            raise RuntimeError("Must have save_ao = True")

        # draw in new fig
        plt.figure()
        ax = plt.gca()

        # get data from lists
        ao_data = np.hstack(self._output_voltages)

        # time stamps
        x = np.arange(len(ao_data[0]))/self._clock_freq

        # draw
        for i, ch in enumerate(self.ao):
            ax.plot(x, ao_data[i], ls='--', label=f'ao{ch}',
                    **plot_kw)

        # plot elements
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage Readback')
        plt.legend(fontsize='x-small')
        plt.tight_layout()

    def reset_filters(self):
        """Reset all signal filters"""
        self.signal_filters = []
        self.do_filter = False

    def run(self, duration, draw_s=0, sample_freq=None, save_ao=False, draw_ch_top=None):
        """Take data, inputs are sine parameters

        Args:
            duration (int): run duration in seconds
            draw_s (int):   if > 0, draw while in progress the last self._draw_s seconds
            sample_freq (float): frequency of sampling in Hz (down-sampled in software from clock_freq). If None, sample at clock_freq.
            save_ao (bool): if true, save signal output for later draw. May crash the run
                            if too long, very memory intensive and append gets slow at long list lengths
            draw_ch_top (str): which channel to draw on top of the others, can be partial name

        Returns:
            None, saves captured data as self.df
        """

        # run setup
        self._setup()

        # save inputs
        self._save_ao = save_ao
        self._draw_s = int(draw_s) if draw_s > 0 else 0

        duration = int(duration)

        if sample_freq is None:
            sample_freq = self._clock_freq

        # get generator for output signal
        self.signal_generator = {ch: self._make_signal_generator(fn) for ch, fn in self.ao.items()}

        # data to output to ao
        self._output_voltages = []

        # set up buffer to read
        self._buffer_in = np.zeros((self._len_ai, self._samples_per_channel))

        # downsampling parameters
        self._downsample = int(self._clock_freq / sample_freq)

        # length after downsample (only if not filtering)
        if self.do_filter:
            self._len_buffer = int(np.ceil(self._samples_per_channel))

        else:
            self._len_buffer = int(np.ceil(self._samples_per_channel / self._downsample))

            # check downsample
            if self._downsample > self._samples_per_channel:
                raise RuntimeError('Buffer size too small for such a slow sampling frequency. '+\
                                'Increase samples_per_channel or sample_freq')

        # calculate total length of output array
        if self.do_filter:
            total_len = duration * self._clock_freq
        else:
            total_len = duration * sample_freq
        nbuffers_toread = int(np.ceil(total_len / self._len_buffer))

        # ensure that length of final data is set properly - account for rounding errors
        total_len = self._len_buffer*nbuffers_toread

        # data to save from ai, final output
        self._data = np.zeros((self._len_ai, total_len))-1111 # -1111 to easily detect default fill data
        self._time = np.zeros((total_len))-1111 # -1111 to easily detect default fill data
        self._nbuffers_read = 0  # number of buffers read out
        self._total_pts = 0 # number of points read before downsample

        # number of points to draw
        ndraw = self._draw_s*sample_freq
        xdata = -1*np.arange(ndraw)/sample_freq
        xdata = xdata[::-1]
        self._ydata = np.zeros((self._len_ai, ndraw))

        # initial fill of empty buffer (required else error)
        for _ in range(self._frames_per_buffer):
            self._write_task_callback(None, None, None, None)

        # start figure for drawing
        if self._draw_s > 0:
            plt.figure()
            self._ax = plt.gca()
            self._ax.set_xlabel('Time (s)')
            self._ax.set_ylabel('Voltage (V)')
            self._ax.set_xlim(-self._draw_s, 0)

            for d, ch in zip(self._ydata, self.ai.values()):

                # set draw order
                if (draw_ch_top is not None) and ch in draw_ch_top:
                    zorder = 20
                else:
                    zorder = 1
                self._ax.plot(xdata, d, label=ch, zorder=zorder)

            self._ax.figure.legend(fontsize='xx-small')
            self._ax.figure.canvas.draw()
            plt.pause(0.1)
            plt.tight_layout()
            self._ax.figure.canvas.manager.window.attributes('-topmost', False)

        # start tasks (begin run)
        self._taski.start()
        self._tasko.start()

        # setup progress bar
        time_start = time.time()
        dt = time_start - time.time()
        time_current = time_start

        progress_bar = tqdm(total=duration, leave=False, unit_scale=True,
                            miniters=1)

        # setup draw counter
        nbuffers_read = 0
        try:

            # progress bar and update figure
            while self._nbuffers_read < nbuffers_toread:

                # progress
                time_prev = time_current
                time_current = time.time()
                dt = float(f'{time_current - time_prev:.2f}')
                progress_bar.update(dt)

                # update figure
                if self._draw_s > 0 and self._nbuffers_read > nbuffers_read:
                    if not self._draw_in_progress():
                        break
                    nbuffers_read = self._nbuffers_read

        # if error, close task nicely
        except Exception as err:
            self.close()
            progress_bar.close()
            raise err from None

        progress_bar.close()

        # set output to zero and stop task
        zeros = np.zeros((self._len_ao, self._samples_per_frame))
        for _ in range(self._frames_per_buffer):
            self._stream_out.write_many_sample(zeros, timeout=1)
        self.close()

        # reassign data to dataframe
        self.df = pd.DataFrame({chname:self._data[i] for i, chname in enumerate(self.ai.values())}, index=self._time)

        self.df.index /= self._clock_freq
        self.df.index.name = 'time (s)'

    def set_filter(self, low=None, high=None, order=6, bandstop=False):
        """Make a butterworth filter which is applied to the readback signal prior to downsampling.
        If low only: low pass filter. If high only, high pass filter. If both, band pass filter.

        Args:
            low (float): lower bound cutoff frequency in Hz
            high (float): upper bound cutoff frequency in Hz
            order (int): filter order
            bandstop (bool): if True, set bandstop instead of bandpass if both low and high set

        Returns
            None, saved to self.signal_filters
        """

        # update limits
        nyq = 0.5 * self._clock_freq
        filter_type = None

        if low is not None:
            low = low / nyq
            filter_type = 'lowpass'
            Wn = low

        if high is not None:
            high = high / nyq

            if filter_type is None:
                filter_type = 'highpass'
                Wn = high
            else:
                filter_type = 'bandpass'
                Wn = (low, high)

        # check if bandstop
        if filter_type == 'bandpass' and bandstop:
           filter_type = 'bandstop'

        # set filter (applied as y = sosfilt(sos, y))
        self.signal_filters.append(butter(order, Wn, btype=filter_type, analog=False, output='sos'))
        self.do_filter = True

    def setup(self, ao = None, ai = None):
        """Set up channels for input and output

        Args:
            ao (dict): connections to analog outputs. Format as ch:(function_handle), where "ch" is an int starting from 0, and "function_handle" is a function handle of the format voltage = fn(time)
            ai (dict): connections to analog inputs. Format as ch:connection, where "ch" is an int starting from 0, and "connection" is a label specifying what the channel is connected to.
        Returns:
            None
        """

        # set channels
        self.ai = ai
        self.ao = ao

    def to_csv(self, path=None, do_filtering=True, do_downsample=True, header=None):
        """Write data to file specified by path

        Args:
            path (str): if None, generate default filename (nidaq_usb6281_yymmddThhmmss.csv), else write csv to this path
            do_filter (bool): if true, if there is a filter, then apply it before saving
            do_downsample (bool): if true, apply downsampling before saving (if no filtering, this option forced True)
            header (str): header string. Note that file written and datetime are appended. Use "#" as comment indicator

        Returns:
            None, writes to file
        """

        # filter and downsample data
        data = self.df.copy()
        if self.do_filter and do_filtering:
            for c in data.columns:
                data.loc[:, c] = self._filter(data.loc[:, c])

        if self.do_filter and do_downsample:
            data = data.loc[::self._downsample]

        # generate default filename
        if path is None:
            path= ['nidaq_usb6281_',
                        datetime.now().strftime('%y%m%dT%H%M%S'),
                        '.csv']
            path = ''.join(path)

        # ensure filename format
        path = os.path.splitext(path)[0]
        path = os.path.abspath(path + '.csv')

        # format physical notes
        if header is None:
            header = ''

        try:
            thisfile = __main__.__file__
        except NameError:
            thisfile = 'USB6281.py'

        # header lines
        header = [  header.strip(),
                    f'# File written by: {thisfile}',
                    f'# {str(datetime.now())}',
                    '# \n']
        with open(path, 'w') as fid:
            fid.write('\n'.join(header))

        # write data to file
        data.to_csv(path, mode='a', index=True)

        print(f'\nSaved file as {path}')