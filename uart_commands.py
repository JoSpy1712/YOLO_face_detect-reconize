# Document the purpose of this module, class, or function.
"""Newline-delimited ASCII commands and servo state for Pico over UART."""
# Run background workers and protect state shared between threads.
import threading
# Read monotonic clocks and pace loops without depending on wall-clock changes.
import time


# Define UARTCommands to group its related state and methods.
class UARTCommands:
    # Initialize the object and the resources or state it owns.
    def __init__(self, port="/dev/serial0", baudrate=115200):
        # Attempt this operation and handle the expected failures below.
        try:
            # Import serial.
            import serial
        # Handle ImportError without losing required cleanup.
        except ImportError as error:
            # Report RuntimeError('UART requires pyserial: .venv/bin/python -m pip install pyserial').
            raise RuntimeError("UART requires pyserial: .venv/bin/python -m pip install pyserial") from error
        # Set self.connection from serial.Serial(port=port, baudrate=baudrate, timeout=0.1, write_timeout=0.5, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbit....
        self.connection = serial.Serial(
            # Supply port as a named argument to the enclosing call.
            port=port, baudrate=baudrate, timeout=0.1, write_timeout=0.5,
            # Supply bytesize as a named argument to the enclosing call.
            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
            # Supply stopbits as a named argument to the enclosing call.
            stopbits=serial.STOPBITS_ONE,
        # Finish the preceding expression, container, or function call.
        )
        # Set the lock preventing interleaved UART writes from different threads using threading.Lock().
        self._lock = threading.Lock()
        # Set the monotonic deadline for the three-second approval flag using 0.0.
        self._unlock_until = 0.0

    # Encode one newline-terminated ASCII command and reject failed or partial serial writes.
    def _write(self, command):
        # Continue with this branch only when not command or any((c in command for c in '\r\n')).
        if not command or any(c in command for c in "\r\n"):
            # Report ValueError('Command must be a nonempty single line').
            raise ValueError("Command must be a nonempty single line")
        # Set payload from (command + '\n').encode('ascii').
        payload = (command + "\n").encode("ascii")
        # Continue with this branch only when self.connection.write(payload) != len(payload).
        if self.connection.write(payload) != len(payload):
            # Report OSError('UART command was only partially written').
            raise OSError("UART command was only partially written")

    # Serialize a legacy command and start the three-second approval interval after a successful write.
    def send_command(self, command):
        # Enter self._lock and release or restore it automatically on exit.
        with self._lock:
            # Call self._write with the arguments below.
            self._write(command)
            # Continue with this branch only when command == 'APPROVED'.
            if command == "APPROVED":
                # Set the monotonic deadline for the three-second approval flag using time.monotonic() + 3.0.
                self._unlock_until = time.monotonic() + 3.0

    # Validate servo angles and send STATE with the current approval flag under the UART lock.
    def send_state(self, pan, tilt):
        # Set (pan, tilt) from (float(pan), float(tilt)).
        pan, tilt = float(pan), float(tilt)
        # Continue with this branch only when not (0 <= pan <= 180 and 0 <= tilt <= 180).
        if not (0 <= pan <= 180 and 0 <= tilt <= 180):
            # Report ValueError('servo angles must be within 0..180').
            raise ValueError("servo angles must be within 0..180")
        # Enter self._lock and release or restore it automatically on exit.
        with self._lock:
            # Set the current LED approval flag, not a physical lock output using time.monotonic() < self._unlock_until.
            unlock = time.monotonic() < self._unlock_until
            # Call self._write with the arguments below.
            self._write("STATE,{:.2f},{:.2f},{}".format(pan, tilt, int(unlock)))

    # Shut down this object and release its associated resources.
    def close(self):
        # Enter self._lock and release or restore it automatically on exit.
        with self._lock:
            # Release the serial device so another program can open it.
            self.connection.close()
