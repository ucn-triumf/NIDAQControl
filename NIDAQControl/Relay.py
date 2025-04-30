# control relay box
# Derek Fujimoto
# Jan 2024

import nidaqmx as ni
import time
import numpy as np

class Relay():
    """Control the switches on the relay box via NIDAQ
    """
    device_name = ["Dev1/port0/line0:7", # [1, 8]
                   "Dev1/port1/line0:7", # [9, 16]
                   "Dev1/port2/line0:7", # [17, 24]
                  ]

    total_ch = 24

    def __init__(self):

        # initial channel state is to disconnect everything
        self.value = np.full(self.total_ch, True)

    def _switch(self, ch, disconnect=True):
        """connect or disconnect a channel

        Args:
            ch (int|list):  if int: channel number, if 0 (dis)connnect all
                            if list: connect according to this scheme (True = connect)
            disconnect (bool): if true, disconnect the switch
        """

        # save old configuration
        old_value = np.copy(self.value)

        # write a configuration
        if type(ch) in (tuple, list, np.ndarray):
            for i in ch:
                self.value[i-1] = disconnect

        # single channel
        else:
            # index by zero
            ch -= 1

            # do all channels and return
            if ch == -1:
                self.value = np.full(self.total_ch, True)

            # write single channel to NIDAQ
            else:
                self.value[ch] = disconnect

        # do the switch
        try:
            self._write()

        # reset channel states if bad connection
        except ChannelCountError as err:
            self.value = old_value
            raise err from None

    def _write(self):
        """Write self.value to nidaq"""

        # check configuration
        if sum(~self.value) > 8:
            raise ChannelCountError("Relay box cannot accept more than 8 channels simultaneously connected at once (likely related to power supply)")

        # create task to connect to NIDAQ
        task = ni.Task()

        # connect to digital output channels
        for device in self.device_name:
            task.do_channels.add_do_chan(device,
                         line_grouping=ni.constants.LineGrouping.CHAN_PER_LINE)

        # write a configuration
        task.write(self.value)
        task.close()
        time.sleep(0.1)

    def connect(self, ch):       self._switch(ch,    disconnect=False)
    def connect_all(self):       self._switch(0,     disconnect=False)
    def disconnect(self, ch):    self._switch(ch,    disconnect=True)
    def disconnect_all(self):    self._switch(0,     disconnect=True)

# exceptions
class ChannelCountError(Exception): pass