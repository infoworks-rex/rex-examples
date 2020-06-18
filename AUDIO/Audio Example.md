# Audio Example

## arecord, aplay

### Options

일부 옵션 생략됨

- **-h, –help :** Show the help information.
- **–version :** Print current version.
- **-l, –list-devices :** List all soundcards and digital audio devices.
- **-L, –list-pcms :** List all PCMs(Pulse Code Modulation) defined.
- **-D, –device=NAME :** Select PCM by name.
- **-q –quiet :** Quiet mode. Suppress messages (not sound :)).
- **-t, –file-type TYPE :** File type (voc, wav, raw or au). If this parameter is omitted the WAVE format is used.
- **-c, –channels=# :** The number of channels. The default is one channel. Valid values are 1 through 32.
- **-f –format=FORMAT :** If no format is given U8 is used.
- **-r, –rate=# :** Sampling rate in Hertz. The default rate is 8000 Hertz.
- **-d, –duration=# :** Interrupt after # seconds.
- **-s, –sleep-min=# :** Min ticks to sleep. The default is not to sleep.
- **-M, –mmap :** Use memory-mapped (mmap) I/O mode for the audio stream. If this option is not set, the read/write I/O mode will be used.
- **-N, –nonblock :** Open the audio device in non-blocking mode. If the device is busy the program will exit immediately.

### Recording Example

#### Default Option

```
[root@Infoworks-REX-basic:/]# arecord default.wav                            
Recording WAVE 'default.wav' : Unsigned 8 bit, Rate 8000 Hz, Mono
```

기본 설정인 wav, unsigned 8-bit, 8000Hz, Mono로 녹음. Ctrl+C 입력으로 중단.

#### CD Quality

```
[root@Infoworks-REX-basic:/]# arecord -r 44100 -f S16_LE -c 2 cd-quality.wav
Recording WAVE 'cd-quality.wav' : Signed 16 bit Little Endian, Rate 44100 Hz, Stereo
```

44.1KHz, Signed 16-bit(Little Endian), Stereo 설정 후 녹음. Ctrl+C 입력으로 중단.

### Playing Example

```
[root@Infoworks-REX-basic:/]# aplay default.wav
Playing WAVE 'default.wav' : Unsigned 8 bit, Rate 8000 Hz, Mono
```

```
[root@Infoworks-REX-basic:/]# aplay cd-quality.wav 
Playing WAVE 'cd-quality.wav' : Signed 16 bit Little Endian, Rate 44100 Hz, Stereo
```

aplay는 파일에서 header를 읽어 data width와 sample rate, channel 수 등을 자동으로 설정.