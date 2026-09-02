# Inter-process protocol

The desktop application never touches the game process directly. It publishes
recognised spells into Windows shared memory, and an in-game component consumes
them. There are two such channels, one per in-game component. Both use the same
lock-free single-producer / single-consumer ring buffer design.

| Channel | Mapping name | Producer | Consumer | Payload |
|---------|--------------|----------|----------|---------|
| Spell names | `SpellCasterSharedMemory` | `SpellTransmitter` (app) | UE4SS mod | 32-byte `SpellCommand` |
| VR controller input | `UEVRSpellCaster` | `VRSharedMemoryServer` (app) | UEVR plugin | 16-byte `VRCommand` |

The definitions live in `app/Whisper-Spell/SpellTransmitter.h`,
`ue4ss-mod/SpellCasterMod/dllmain.cpp` (namespace `SharedMemory`),
`app/Whisper-Spell/SharedMemoryStructure.h` and
`uevr-plugin/UEVRSpellCasterPlugin/UEVRSpellCasterPluginSharedMemory.cpp`.
The structures are duplicated on each side and must be kept byte-identical.
`ue4ss-mod/tools/spell_sender.py` is a reference producer for the spell channel.

## Spell channel layout

All fields are little-endian 32-bit unsigned integers unless stated otherwise.
The mapping is created in the session-local namespace with `PAGE_READWRITE`
access; whichever side starts first creates it, the other one opens it
(`OpenFileMappingA`, falling back to `CreateFileMappingA`).

```
offset  size   field
------  -----  -----------------------------------------------------------
0x0000  4      writeIndex          producer's next slot (monotonic)
0x0004  4      readIndex           consumer's next slot (monotonic)
0x0008  4      producerHeartbeat   incremented periodically by the app
0x000C  4      consumerHeartbeat   incremented every 5 frames by the mod
0x0010  4      messagesDropped     producer increments when the ring is full
0x0014  4      totalMessages       producer increments on every write
0x0018  40     padding             reserved, keeps the control block at 64 bytes

0x0040  8192   commandBuffer[256]  ring of SpellCommand (32 bytes each)
          +0   char spellName[28]  NUL-terminated ASCII
          +28  uint32 flags        reserved, always 0 today

0x2040  4      statusWriteIndex    reserved for consumer -> producer feedback
0x2044  4      statusReadIndex
0x2048  4088   statusBuffer        reserved, unused today

total   0x3040 (12 352 bytes)
```

### Rules

- Indices only grow. The slot of an index is `index & 255`.
- The ring is full when `((writeIndex + 1) & 255) == (readIndex & 255)`. A
  producer that finds it full drops the message and increments
  `messagesDropped`; it never overwrites unread slots.
- The producer fills the slot, then publishes `writeIndex + 1` with release
  semantics. The consumer loads `writeIndex` with acquire semantics, reads the
  slot, then publishes `readIndex + 1`.
- Heartbeats are the only liveness signal. The application reports the mod as
  disconnected when `consumerHeartbeat` has not changed for two seconds
  (`SpellTransmitter::CONSUMER_TIMEOUT_MS`); the mod only displays
  `producerHeartbeat` in its queue status.
- `spellName` is the spell's display name in PascalCase without spaces,
  punctuation or accents, at most 27 characters. The app derives it from the
  transcription (`SpellTransmitter::normalizeSpellName`), so "avada kedavra"
  becomes `AvadaKedavra`. The mod matches names case-insensitively.
- Unknown spells are consumed and discarded by the mod; a spell whose tool is
  temporarily unavailable stays in the mod's internal queue and is retried on
  the following frames.

### Versioning

There is no version field yet. `flags` is the place to add one; until then,
any change to the layout must be applied to every implementation listed above
and released together.

## VR channel layout

Same control block (64 bytes, with the heartbeats named `pluginHeartbeat` and
`mainAppHeartbeat`), followed by a ring of 256 `VRCommand` entries of 16 bytes
and the same 4 KB status area, for a total of 8 256 bytes.

```
VRCommand
  +0   uint8   type              1 = single press, 2 = combo (sequence),
                                 3 = principal (right trigger + buttons),
                                 4 = simultaneous
  +1   uint8   buttonCount       1 to 4
  +2   uint16  duration          hold time in milliseconds
  +4   uint16  buttonSequence[4] XUSB button constants, 0 = unused
  +12  uint32  reserved
```

The UEVR plugin merges these presses into the XInput state it reports to the
game, on top of whatever the physical controller is doing.
