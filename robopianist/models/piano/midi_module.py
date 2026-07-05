# Copyright 2023 The RoboPianist Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Piano sound module."""

from typing import Callable, List, Optional

import numpy as np
from dm_control import mjcf

from robopianist.models.piano import piano_constants
from robopianist.music import midi_file, midi_message

# Linear key-speed -> MIDI velocity mapping. qvel is the key hinge angular
# velocity (rad/s) at the substep the key crosses the activation threshold,
# the simulated analogue of a MIDI keyboard's second sensor contact. The
# synth's velocity->attenuation curve is logarithmic (SoundFont spec), so a
# linear mapping here reproduces the acoustic SPL ∝ log(key speed) behavior.
# Bounds calibrated empirically (symphony scripts/calibrate_velocity.py).
_QVEL_VMIN = 0.73
# VMAX is a sensitivity setting (like a keyboard's velocity curve), not a
# physical limit. The initial 15.1 (99th pct incl. random flailing) put forte
# out of reach of accurate playing; 8.0 maps the policy's controllable range
# (~0.7-8 rad/s) onto the full 1-127.
_QVEL_VMAX = 8.0
# Soft velocity curve (gamma < 1), the standard "light touch" keyboard
# setting: lifts gentle presses into the audible range. With the synth's
# 40*log10(v/127) attenuation, gamma=0.5 keeps intentional pp presses above
# ~-30 dB instead of vanishing at -45 dB.
_CURVE_GAMMA = 0.5


def qvel_to_midi_velocity(qvel: float) -> int:
    frac = (qvel - _QVEL_VMIN) / (_QVEL_VMAX - _QVEL_VMIN)
    frac = np.clip(frac, 0.0, 1.0) ** _CURVE_GAMMA
    return int(np.clip(np.round(1 + 126.0 * frac), 1, 127))


class MidiModule:
    """The piano sound module.

    It is responsible for tracking the state of the piano keys and generating
    corresponding MIDI messages. The MIDI messages can be used with a synthesizer
    to produce sound.
    """

    def __init__(self) -> None:
        self._note_on_callback: Optional[Callable[[int, int], None]] = None
        self._note_off_callback: Optional[Callable[[int], None]] = None
        self._sustain_on_callback: Optional[Callable[[], None]] = None
        self._sustain_off_callback: Optional[Callable[[], None]] = None

    def initialize_episode(self, physics: mjcf.Physics) -> None:
        del physics  # Unused.

        self._prev_activation = np.zeros(piano_constants.NUM_KEYS, dtype=bool)
        self._prev_sustain_activation = np.zeros(1, dtype=bool)
        self._midi_messages: List[List[midi_message.MidiMessage]] = []

    def after_substep(
        self,
        physics: mjcf.Physics,
        activation: np.ndarray,
        sustain_activation: np.ndarray,
        key_qvel: Optional[np.ndarray] = None,
    ) -> None:
        # Sanity check dtype since we use bitwise operators.
        assert activation.dtype == bool
        assert sustain_activation.dtype == bool

        timestep_events: List[midi_message.MidiMessage] = []
        message: midi_message.MidiMessage

        state_change = activation ^ self._prev_activation
        sustain_change = sustain_activation ^ self._prev_sustain_activation

        # Note on events.
        for key_id in np.flatnonzero(state_change & ~self._prev_activation):
            message = midi_message.NoteOn(
                note=midi_file.key_number_to_midi_number(key_id),
                velocity=qvel_to_midi_velocity(key_qvel[key_id])
                if key_qvel is not None
                else 127,
                time=physics.data.time,
            )
            timestep_events.append(message)
            if self._note_on_callback is not None:
                self._note_on_callback(message.note, message.velocity)

        # Note off events.
        for key_id in np.flatnonzero(state_change & ~activation):
            message = midi_message.NoteOff(
                note=midi_file.key_number_to_midi_number(key_id),
                time=physics.data.time,
            )
            timestep_events.append(message)
            if self._note_off_callback is not None:
                self._note_off_callback(message.note)

        # Sustain pedal events.
        if sustain_change & ~self._prev_sustain_activation:
            timestep_events.append(midi_message.SustainOn(time=physics.data.time))
            if self._sustain_on_callback is not None:
                self._sustain_on_callback()
        if sustain_change & ~sustain_activation:
            timestep_events.append(midi_message.SustainOff(time=physics.data.time))
            if self._sustain_off_callback is not None:
                self._sustain_off_callback()

        self._midi_messages.append(timestep_events)
        self._prev_activation = activation.copy()
        self._prev_sustain_activation = sustain_activation.copy()

    def get_latest_midi_messages(self) -> List[midi_message.MidiMessage]:
        """Returns the MIDI messages generated in the last substep."""
        return self._midi_messages[-1]

    def get_all_midi_messages(self) -> List[midi_message.MidiMessage]:
        """Returns a list of all MIDI messages generated during the episode."""
        return [message for timestep in self._midi_messages for message in timestep]

    # Callbacks for synthesizer events.

    def register_synth_note_on_callback(
        self,
        callback: Callable[[int, int], None],
    ) -> None:
        """Registers a callback for note on events."""
        self._note_on_callback = callback

    def register_synth_note_off_callback(
        self,
        callback: Callable[[int], None],
    ) -> None:
        """Registers a callback for note off events."""
        self._note_off_callback = callback

    def register_synth_sustain_on_callback(
        self,
        callback: Callable[[], None],
    ) -> None:
        """Registers a callback for sustain pedal on events."""
        self._sustain_on_callback = callback

    def register_synth_sustain_off_callback(
        self,
        callback: Callable[[], None],
    ) -> None:
        """Registers a callback for sustain pedal off events."""
        self._sustain_off_callback = callback
