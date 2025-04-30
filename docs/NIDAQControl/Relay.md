# Relay

[Nidaqcontrol Index](../README.md#nidaqcontrol-index) / [Nidaqcontrol](./index.md#nidaqcontrol) / Relay

> Auto-generated documentation for [NIDAQControl.Relay](../../NIDAQControl/Relay.py) module.

- [Relay](#relay)
  - [ChannelCountError](#channelcounterror)
  - [Relay](#relay-1)
    - [Relay.\_switch](#relay_switch)
    - [Relay.\_write](#relay_write)
    - [Relay.connect](#relayconnect)
    - [Relay.connect\_all](#relayconnect_all)
    - [Relay.disconnect](#relaydisconnect)
    - [Relay.disconnect\_all](#relaydisconnect_all)

## ChannelCountError

[Show source in Relay.py:89](../../NIDAQControl/Relay.py#L89)

#### Signature

```python
class ChannelCountError(Exception): ...
```



## Relay

[Show source in Relay.py:9](../../NIDAQControl/Relay.py#L9)

Control the switches on the relay box via NIDAQ

#### Signature

```python
class Relay:
    def __init__(self): ...
```

### Relay._switch

[Show source in Relay.py:24](../../NIDAQControl/Relay.py#L24)

connect or disconnect a channel

#### Arguments

- `ch` *int|list* - if int: channel number, if 0 (dis)connnect all
                - `if` *list* - connect according to this scheme (True = connect)
- [Relay.disconnect](#relaydisconnect) *bool* - if true, disconnect the switch

#### Signature

```python
def _switch(self, ch, disconnect=True): ...
```

### Relay._write

[Show source in Relay.py:63](../../NIDAQControl/Relay.py#L63)

Write self.value to nidaq

#### Signature

```python
def _write(self): ...
```

### Relay.connect

[Show source in Relay.py:83](../../NIDAQControl/Relay.py#L83)

#### Signature

```python
def connect(self, ch): ...
```

### Relay.connect_all

[Show source in Relay.py:84](../../NIDAQControl/Relay.py#L84)

#### Signature

```python
def connect_all(self): ...
```

### Relay.disconnect

[Show source in Relay.py:85](../../NIDAQControl/Relay.py#L85)

#### Signature

```python
def disconnect(self, ch): ...
```

### Relay.disconnect_all

[Show source in Relay.py:86](../../NIDAQControl/Relay.py#L86)

#### Signature

```python
def disconnect_all(self): ...
```