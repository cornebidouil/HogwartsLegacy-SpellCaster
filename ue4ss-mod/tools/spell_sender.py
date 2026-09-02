#!/usr/bin/env python3
"""
Spell Sender - Shared memory client for the SpellCaster UE4SS mod
Sends spell commands to Hogwarts Legacy via shared memory.

Usage:
    # Interactive mode
    python spell_sender.py

    # Single spell (CLI mode)
    python spell_sender.py Lumos
    python spell_sender.py AppertaCodex

    # Multiple spells (combo)
    python spell_sender.py Lumos Incendio Stupefy

Then type spell names like: Lumos, Incendio, Revelio, AvadaKedavra
Or menu spells: AppertaCodex, AppertaMappa, Finite
Type 'quit' or 'exit' to close.
Type 'list' to see available spells.
"""

import mmap
import struct
import time
import sys
import argparse

# Shared memory configuration (must match C++ code)
SHARED_MEMORY_NAME = "SpellCasterSharedMemory"
COMMAND_BUFFER_SIZE = 256
COMMAND_BUFFER_MASK = COMMAND_BUFFER_SIZE - 1

# Structure sizes
CONTROL_SIZE = 64  # Control structure (64 bytes)
COMMAND_SIZE = 32  # Each SpellCommand (28 bytes name + 4 bytes flags)
STATUS_SIZE = 4096  # Status area

TOTAL_SIZE = CONTROL_SIZE + (COMMAND_BUFFER_SIZE * COMMAND_SIZE) + STATUS_SIZE

# Control structure offsets
OFFSET_WRITE_INDEX = 0
OFFSET_READ_INDEX = 4
OFFSET_PRODUCER_HEARTBEAT = 8
OFFSET_CONSUMER_HEARTBEAT = 12
OFFSET_MESSAGES_DROPPED = 16
OFFSET_TOTAL_MESSAGES = 20

# Command buffer starts after control structure
OFFSET_COMMAND_BUFFER = CONTROL_SIZE

# Available spells (for reference)
AVAILABLE_SPELLS = [
    # Menu Access (Custom Latin Spells)
    "AppertaCodex", "AppertaSacculus", "AppertaMeritas", "AppertaVestiarium",
    "AppertaFalcultates", "AppertaIncantatem", "AppertaLiterae", "AppertaMappa",
    "AppertaCompendium", "AppertaConfiguratio", "AppertaQuaestiones", "Finite",
    # Navigation
    "AppareVestigium",  # Show objective path (V key)
    # Mount/Broom
    "AccioBroomstick", "AccioBalais", "AccioFirebolt", "AccioEclairDeFeu",  # Summon broomstick
    "AccioMount", "AccioMonture", "AccioHippogriff", "AccioHippogriffe",  # Summon creature mount
    "AccioGraphorn", "AccioSombral", "AccioThestral",  # Specific mounts
    # Control
    "Accio", "Levioso", "Depulso", "Descendo", "Flipendo", "Glacius", "ArrestoMomentum",
    # Damage
    "Incendio", "Confringo", "Diffindo", "Expulso", "Bombarda",
    # Combat
    "Stupefy", "Expelliarmus", "Protego", "Oppugno",
    # Utility
    "Lumos", "Reparo", "Revelio", "Disillusionment", "Wingardium",
    # Transfiguration
    "Transformation", "Conjuration", "Vanishment",
    # Unforgivable
    "AvadaKedavra", "Crucio", "Imperius",
    # Special
    "Finisher", "StealthTakedown", "Confundo", "Episkey",
]


class SpellSender:
    def __init__(self):
        self.shm = None
        self.mm = None

    def connect(self):
        """Connect to or create the shared memory."""
        try:
            # Try to open existing shared memory
            self.shm = mmap.mmap(-1, TOTAL_SIZE, tagname=SHARED_MEMORY_NAME, access=mmap.ACCESS_WRITE)
            print(f"[OK] Connected to shared memory: {SHARED_MEMORY_NAME}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to connect to shared memory: {e}")
            return False

    def disconnect(self):
        """Disconnect from shared memory."""
        if self.shm:
            self.shm.close()
            self.shm = None
            print("[OK] Disconnected from shared memory")

    def read_uint32(self, offset):
        """Read a 32-bit unsigned integer from shared memory."""
        self.shm.seek(offset)
        data = self.shm.read(4)
        return struct.unpack('<I', data)[0]

    def write_uint32(self, offset, value):
        """Write a 32-bit unsigned integer to shared memory."""
        self.shm.seek(offset)
        self.shm.write(struct.pack('<I', value))

    def update_heartbeat(self):
        """Update producer heartbeat to signal we're alive."""
        current = self.read_uint32(OFFSET_PRODUCER_HEARTBEAT)
        self.write_uint32(OFFSET_PRODUCER_HEARTBEAT, current + 1)

    def get_status(self):
        """Get current shared memory status."""
        write_idx = self.read_uint32(OFFSET_WRITE_INDEX)
        read_idx = self.read_uint32(OFFSET_READ_INDEX)
        producer_hb = self.read_uint32(OFFSET_PRODUCER_HEARTBEAT)
        consumer_hb = self.read_uint32(OFFSET_CONSUMER_HEARTBEAT)
        dropped = self.read_uint32(OFFSET_MESSAGES_DROPPED)
        total = self.read_uint32(OFFSET_TOTAL_MESSAGES)

        pending = (write_idx - read_idx) & 0xFFFFFFFF

        return {
            'write_index': write_idx,
            'read_index': read_idx,
            'pending': pending,
            'producer_heartbeat': producer_hb,
            'consumer_heartbeat': consumer_hb,
            'messages_dropped': dropped,
            'total_messages': total,
        }

    def send_spell(self, spell_name):
        """Send a spell command to the shared memory buffer."""
        if not self.shm:
            print("[ERROR] Not connected to shared memory")
            return False

        # Validate spell name length
        if len(spell_name) > 27:
            print(f"[ERROR] Spell name too long (max 27 chars): {spell_name}")
            return False

        # Get current write position
        write_idx = self.read_uint32(OFFSET_WRITE_INDEX)
        read_idx = self.read_uint32(OFFSET_READ_INDEX)

        # Check if buffer is full
        next_write = (write_idx + 1) & 0xFFFFFFFF
        if (next_write & COMMAND_BUFFER_MASK) == (read_idx & COMMAND_BUFFER_MASK):
            # Buffer full - increment dropped counter
            dropped = self.read_uint32(OFFSET_MESSAGES_DROPPED)
            self.write_uint32(OFFSET_MESSAGES_DROPPED, dropped + 1)
            print(f"[WARNING] Buffer full, spell dropped: {spell_name}")
            return False

        # Calculate buffer position
        buffer_index = write_idx & COMMAND_BUFFER_MASK
        cmd_offset = OFFSET_COMMAND_BUFFER + (buffer_index * COMMAND_SIZE)

        # Prepare command data (28 bytes name + 4 bytes flags)
        spell_bytes = spell_name.encode('ascii')[:27]
        spell_bytes = spell_bytes.ljust(28, b'\x00')  # Pad with nulls
        flags = struct.pack('<I', 0)  # Flags = 0

        # Write command to buffer
        self.shm.seek(cmd_offset)
        self.shm.write(spell_bytes + flags)

        # Update write index (this makes the command visible to consumer)
        self.write_uint32(OFFSET_WRITE_INDEX, next_write)

        # Update total messages counter
        total = self.read_uint32(OFFSET_TOTAL_MESSAGES)
        self.write_uint32(OFFSET_TOTAL_MESSAGES, total + 1)

        # Update heartbeat
        self.update_heartbeat()

        print(f"[SENT] {spell_name}")
        return True

    def send_multiple(self, spells, delay=0.1):
        """Send multiple spells with a delay between each."""
        for spell in spells:
            self.send_spell(spell)
            if delay > 0:
                time.sleep(delay)


def print_help():
    """Print help information."""
    print("\n=== Spell Sender Commands ===")
    print("  <spell_name>  - Send a spell (e.g., 'Lumos', 'Incendio')")
    print("  list          - List all available spells")
    print("  status        - Show shared memory status")
    print("  combo <s1> <s2> ... - Send multiple spells")
    print("  quit/exit     - Exit the program")
    print("  help          - Show this help\n")


def print_spell_list():
    """Print list of available spells."""
    print("\n=== Available Spells ===")

    categories = {
        "Menu Access": [
            "AppertaCodex (Inventory)", "AppertaSacculus (Inventory)",
            "AppertaMeritas (Character)", "AppertaVestiarium (Gear)",
            "AppertaFalcultates (Talents)", "AppertaIncantatem (Studies)",
            "AppertaLiterae (Mail)", "AppertaMappa (Map)",
            "AppertaCompendium (Collections)", "AppertaConfiguratio (Settings)",
            "AppertaQuaestiones (Quests)", "Finite (Close Menu)"
        ],
        "Navigation": [
            "AppareVestigium (Show Objective Path)"
        ],
        "Mount": [
            "AccioBroomstick (Summon Broom)",
            "AccioBalais (Summon Broom - French)",
            "AccioFirebolt (Summon Broom)",
            "AccioEclairDeFeu (Summon Broom - French)",
            "AccioMount (Summon Creature Mount)",
            "AccioMonture (Summon Mount - French)",
            "AccioHippogriff (Summon Hippogriff)",
            "AccioHippogriffe (Summon Hippogriff - French)",
            "AccioGraphorn (Summon Graphorn)",
            "AccioThestral (Summon Thestral)",
            "AccioSombral (Summon Thestral - French)"
        ],
        "Control": ["Accio", "Levioso", "Depulso", "Descendo", "Flipendo", "Glacius", "ArrestoMomentum"],
        "Damage": ["Incendio", "Confringo", "Diffindo", "Expulso", "Bombarda"],
        "Combat": ["Stupefy", "Expelliarmus", "Protego", "Oppugno"],
        "Utility": ["Lumos", "Reparo", "Revelio", "Disillusionment", "Wingardium"],
        "Transfiguration": ["Transformation", "Conjuration", "Vanishment"],
        "Unforgivable": ["AvadaKedavra", "Crucio", "Imperius"],
        "Special": ["Finisher", "StealthTakedown", "Confundo", "Episkey"],
    }

    for category, spells in categories.items():
        print(f"\n  {category}:")
        for spell in spells:
            print(f"    {spell}")
    print()


def send_spell_cli(spell_name):
    """Send a single spell in CLI mode (non-interactive)."""
    sender = SpellSender()

    if not sender.connect():
        print("\n[ERROR] Could not connect to shared memory.")
        print("Make sure the game is running with the SpellCaster mod loaded and a save opened.")
        return False

    try:
        # Try to match case-insensitively with known spells
        matched = None
        for known in AVAILABLE_SPELLS:
            if known.lower() == spell_name.lower():
                matched = known
                break

        if matched:
            result = sender.send_spell(matched)
        else:
            # Send as-is (might be a custom/modded spell)
            print(f"[NOTE] Unknown spell '{spell_name}', sending anyway...")
            result = sender.send_spell(spell_name)

        return result

    finally:
        sender.disconnect()


def main():
    # Check if spell names were provided as CLI arguments
    if len(sys.argv) > 1:
        # CLI mode - send spells and exit
        spells = sys.argv[1:]

        print("=" * 50)
        print("  Spell Sender - CLI Mode")
        print("=" * 50)

        sender = SpellSender()
        if not sender.connect():
            print("\n[ERROR] Could not connect to shared memory.")
            print("Make sure the game is running with the SpellCaster mod loaded and a save opened.")
            return

        try:
            if len(spells) == 1:
                print(f"\n[SENDING] {spells[0]}")
                send_spell_cli(spells[0])
            else:
                print(f"\n[COMBO] Sending {len(spells)} spells...")
                sender.send_multiple(spells, delay=0.3)
        finally:
            sender.disconnect()

        return

    # Interactive mode
    print("=" * 50)
    print("  Spell Sender - Hogwarts Legacy Spell Caster")
    print("=" * 50)
    print(f"  Shared Memory: {SHARED_MEMORY_NAME}")
    print("  Type 'help' for commands, 'quit' to exit")
    print("=" * 50)

    sender = SpellSender()

    if not sender.connect():
        print("\n[ERROR] Could not connect to shared memory.")
        print("Make sure the game is running with the SpellCaster mod loaded and a save opened.")
        input("Press Enter to exit...")
        return

    # Show initial status
    status = sender.get_status()
    print(f"\n[STATUS] Consumer heartbeat: {status['consumer_heartbeat']} "
          f"(non-zero = mod is running)")

    try:
        while True:
            try:
                user_input = input("\nSpell> ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd in ('quit', 'exit', 'q'):
                break
            elif cmd == 'help':
                print_help()
            elif cmd == 'list':
                print_spell_list()
            elif cmd == 'status':
                status = sender.get_status()
                print(f"\n[STATUS]")
                print(f"  Write Index: {status['write_index']}")
                print(f"  Read Index: {status['read_index']}")
                print(f"  Pending: {status['pending']}")
                print(f"  Producer Heartbeat: {status['producer_heartbeat']}")
                print(f"  Consumer Heartbeat: {status['consumer_heartbeat']}")
                print(f"  Total Messages: {status['total_messages']}")
                print(f"  Messages Dropped: {status['messages_dropped']}")
            elif cmd == 'combo':
                if len(parts) > 1:
                    spells = parts[1:]
                    print(f"[COMBO] Sending {len(spells)} spells...")
                    sender.send_multiple(spells, delay=0.5)
                else:
                    print("[ERROR] Usage: combo <spell1> <spell2> ...")
            else:
                # Treat as spell name
                spell_name = parts[0]
                # Try to match case-insensitively with known spells
                matched = None
                for known in AVAILABLE_SPELLS:
                    if known.lower() == spell_name.lower():
                        matched = known
                        break

                if matched:
                    sender.send_spell(matched)
                else:
                    # Send as-is (might be a custom/modded spell)
                    print(f"[NOTE] Unknown spell '{spell_name}', sending anyway...")
                    sender.send_spell(spell_name)

    except KeyboardInterrupt:
        print("\n[Interrupted]")

    finally:
        sender.disconnect()
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
