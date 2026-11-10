"""Fixed-rate control loop and a non-blocking "was Enter pressed?" check.

    for t in RealtimeLoop(200):     # t = seconds since the loop started
        ...                         # one control tick
        if t > 5.0:
            break

Ctrl-C raises KeyboardInterrupt as usual; leaving the `with Joint(...)` block then disables the motor.
"""

import select
import sys
import time


class RealtimeLoop:
    """Runs its body `rate_hz` times per second. A tick that runs late is not made up for: the
    next one starts on time again (late ticks are counted in `late_ticks`)."""

    def __init__(self, rate_hz: float) -> None:
        self.dt = 1.0 / rate_hz
        self.start_time = float("nan")  # time.monotonic() at t = 0
        self.late_ticks = 0

    def __iter__(self):
        self.start_time = time.monotonic()
        next_tick = self.start_time
        while True:
            yield time.monotonic() - self.start_time
            next_tick += self.dt
            # Plain sleep, no busy-waiting: a busy loop would hold Python's GIL and starve the CAN
            # receive thread.
            remaining = next_tick - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                self.late_ticks += 1
                next_tick = time.monotonic()


def enter_pressed() -> bool:
    """True if a line was typed on the terminal since the last check (does not wait)."""
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        sys.stdin.readline()
        return True
    return False
