# Document the purpose of this module, class, or function.
"""MicroPython Pico receiver: UART0, GP0 TX / GP1 RX, 115200 8N1.

STATE,pan,tilt,unlock\n plus the existing APPROVED\n command.
"""

# Set the maximum accepted UART line length in bytes using 128.
MAX_LINE = 128
# Set the Pico GPIO used for horizontal servo PWM using 2.
PAN_PIN = 2
# Set the Pico GPIO used for vertical servo PWM using 3.
TILT_PIN = 3
# Set the pulse width assigned to the zero-degree command using 500.
SERVO_MIN_US = 500
# Set the pulse width assigned to the 180-degree command using 2500.
SERVO_MAX_US = 2500


# Validate an ASCII UART command before allowing any LED or servo update.
def parse_command(line):
    # Set text from line.rstrip('\r').
    text = line.rstrip("\r")
    # Continue with this branch only when text == 'APPROVED'.
    if text == "APPROVED":
        # Return (None, None, True) to the caller.
        return None, None, True
    # Set fields from text.split(',').
    fields = text.split(",")
    # Continue with this branch only when len(fields) != 4 or fields[0] != 'STATE' or fields[3] not in ('0', '1').
    if len(fields) != 4 or fields[0] != "STATE" or fields[3] not in ("0", "1"):
        # Report ValueError('expected STATE,pan,tilt,0/1').
        raise ValueError("expected STATE,pan,tilt,0/1")
    # Set (pan, tilt) from (float(fields[1]), float(fields[2])).
    pan, tilt = float(fields[1]), float(fields[2])
    # Continue with this branch only when not (0 <= pan <= 180 and 0 <= tilt <= 180).
    if not (0 <= pan <= 180 and 0 <= tilt <= 180):
        # Report ValueError('angles must be finite and within 0..180').
        raise ValueError("angles must be finite and within 0..180")
    # Return (pan, tilt, fields[3] == '1') to the caller.
    return pan, tilt, fields[3] == "1"


# Map a 0-to-180-degree command to the configured servo pulse width in microseconds.
def servo_pulse_us(angle):
    # Return int(SERVO_MIN_US + (SERVO_MAX_US - SERVO_MIN_US) * angle / 180) to the caller.
    return int(SERVO_MIN_US + (SERVO_MAX_US-SERVO_MIN_US)*angle/180)


# Define LineReceiver to group its related state and methods.
class LineReceiver:
    # Document the purpose of this module, class, or function.
    """Handle split packets, multiple packets, CRLF and oversized lines."""

    # Initialize the object and the resources or state it owns.
    def __init__(self):
        # Set the partial UART line retained until a newline arrives using bytearray().
        self.buffer = bytearray()
        # Set whether bytes must be ignored until the next newline after overflow using False.
        self.discard = False

    # Assemble newline-terminated packets across reads and discard oversized packets safely.
    def feed(self, data):
        # Set lines from [].
        lines = []
        # Process each byte from data.
        for byte in data:
            # Continue with this branch only when byte == 10.
            if byte == 10:
                # Continue with this branch only when not self.discard and self.buffer.
                if not self.discard and self.buffer:
                    # Call lines.append with the arguments below.
                    lines.append(bytes(self.buffer))
                # Set the partial UART line retained until a newline arrives using bytearray().
                self.buffer = bytearray()
                # Set whether bytes must be ignored until the next newline after overflow using False.
                self.discard = False
            # Continue with this branch only when not self.discard.
            elif not self.discard:
                # Continue with this branch only when len(self.buffer) >= MAX_LINE.
                if len(self.buffer) >= MAX_LINE:
                    # Set the partial UART line retained until a newline arrives using bytearray().
                    self.buffer = bytearray()
                    # Set whether bytes must be ignored until the next newline after overflow using True.
                    self.discard = True
                # Handle the remaining case when the preceding conditions did not match.
                else:
                    # Call self.buffer.append with the arguments below.
                    self.buffer.append(byte)
        # Return lines to the caller.
        return lines


# Run this file entry point and manage its resources and cleanup.
def main():
    # Import MicroPython GPIO, hardware serial, and PWM drivers provided by the Pico firmware.
    from machine import Pin, UART, PWM
    # Import millisecond pacing and wrap-safe timestamp arithmetic for LED timeouts and diagnostics.
    from time import sleep_ms, ticks_ms, ticks_diff, ticks_add

    # Attempt this operation and handle the expected failures below.
    try:
        # Set led from Pin('LED', Pin.OUT).
        led = Pin("LED", Pin.OUT)
    # Handle (TypeError, ValueError) without losing required cleanup.
    except (TypeError, ValueError):
        # Set led from Pin(25, Pin.OUT).
        led = Pin(25, Pin.OUT)  # Original non-wireless Pico firmware.
    # Set the onboard LED output to 0.
    led.value(0)
    # Create a hardware PWM output for the selected servo GPIO.
    pan_servo = PWM(Pin(PAN_PIN))
    # Create a hardware PWM output for the selected servo GPIO.
    tilt_servo = PWM(Pin(TILT_PIN))
    # Process each servo from (pan_servo, tilt_servo).
    for servo in (pan_servo, tilt_servo):
        # Set the servo PWM repetition rate to 50 Hz.
        servo.freq(50)
        # Immediately command the initial 90-degree position, even before UART receives data.
        servo.duty_ns(servo_pulse_us(90)*1000)
    # Open Pico UART0 at 115200 baud with 8 data bits, no parity, and one stop bit.
    uart = UART(0, baudrate=115200, bits=8, parity=None, stop=1,
                # Supply tx as a named argument to the enclosing call.
                tx=Pin(0), rx=Pin(1), rxbuf=1024)
    # Set receiver from LineReceiver().
    receiver = LineReceiver()
    # Set the cumulative number of UART bytes received using 0.
    rx_bytes = 0
    # Set the cumulative number of completely validated packets using 0.
    rx_valid = 0
    # Set the cumulative number of rejected packets using 0.
    rx_bad = 0
    # Set the timestamp of the most recent UART bytes using None.
    last_rx = None
    # Set the timestamp of the most recent valid command using None.
    last_valid = None
    # Set the last accepted raw UART line shown in diagnostics using None.
    last_packet = None
    # Set the timestamp used to print UART diagnostics once per second using ticks_ms().
    report_at = ticks_ms()
    # Set the current LED approval flag, not a physical lock output using False.
    unlock = False
    # Set the wrap-safe millisecond deadline after which the approval LED must turn off using ticks_ms().
    led_deadline = ticks_ms()
    # Print a diagnostic, status, or setup message to the console.
    print("Pico ready: UART0 GP1 RX, 115200 baud")
    # Attempt this operation and handle the expected failures below.
    try:
        # Repeat this loop while True.
        while True:
            # Continue with this branch only when uart.any().
            if uart.any():
                # Set data from uart.read(128).
                data = uart.read(128)
                # Continue with this branch only when data.
                if data:
                    # Update the cumulative number of UART bytes received using len(data).
                    rx_bytes += len(data)
                    # Set the timestamp of the most recent UART bytes using ticks_ms().
                    last_rx = ticks_ms()
                # Process each raw from receiver.feed(data or b'').
                for raw in receiver.feed(data or b""):
                    # Attempt this operation and handle the expected failures below.
                    try:
                        # Set (pan, tilt, flag) from parse_command(raw.decode('ascii')).
                        pan, tilt, flag = parse_command(raw.decode("ascii"))
                    # Handle (ValueError, UnicodeError) without losing required cleanup.
                    except (ValueError, UnicodeError) as error:
                        # Update the cumulative number of rejected packets using 1.
                        rx_bad += 1
                        # Continue with this branch only when rx_bad <= 3.
                        if rx_bad <= 3:
                            # Print a diagnostic, status, or setup message to the console.
                            print("Ignored packet:", repr(raw), error)
                        # Skip the rest of this iteration and wait for the next input or frame.
                        continue
                    # Update the cumulative number of completely validated packets using 1.
                    rx_valid += 1
                    # Set the timestamp of the most recent valid command using ticks_ms().
                    last_valid = ticks_ms()
                    # Set the last accepted raw UART line shown in diagnostics using raw.
                    last_packet = raw
                    # Set the current LED approval flag, not a physical lock output using flag.
                    unlock = flag
                    # Continue with this branch only when pan is not None.
                    if pan is not None:
                        # Apply the validated horizontal angle as a PWM pulse width in nanoseconds.
                        pan_servo.duty_ns(servo_pulse_us(pan)*1000)
                        # Apply the validated vertical angle as a PWM pulse width in nanoseconds.
                        tilt_servo.duty_ns(servo_pulse_us(tilt)*1000)
                    # STATE heartbeats refresh a 1 s LED deadline; legacy
                    # APPROVED alone lights it for 3 s. Lost link holds angles.
                    # Set the wrap-safe millisecond deadline after which the approval LED must turn off using ticks_add(ticks_ms(), 3000 if pan is None else 1000).
                    led_deadline = ticks_add(ticks_ms(), 3000 if pan is None else 1000)
                    # Set the onboard LED output to 1 if unlock else 0.
                    led.value(1 if unlock else 0)
            # Continue with this branch only when unlock and ticks_diff(ticks_ms(), led_deadline) >= 0.
            if unlock and ticks_diff(ticks_ms(), led_deadline) >= 0:
                # Set the current LED approval flag, not a physical lock output using False.
                unlock = False
                # Set the onboard LED output to 0.
                led.value(0)
            # Set now from ticks_ms().
            now = ticks_ms()
            # Continue with this branch only when ticks_diff(now, report_at) >= 1000.
            if ticks_diff(now, report_at) >= 1000:
                # Set the timestamp used to print UART diagnostics once per second using now.
                report_at = now
                # Continue with this branch only when last_rx is None.
                if last_rx is None:
                    # Set status from 'NO DATA'.
                    status = "NO DATA"
                # Continue with this branch only when last_valid is not None and ticks_diff(now, last_valid) < 1000.
                elif last_valid is not None and ticks_diff(now, last_valid) < 1000:
                    # Set status from 'RX OK'.
                    status = "RX OK"
                # Continue with this branch only when ticks_diff(now, last_rx) < 1000.
                elif ticks_diff(now, last_rx) < 1000:
                    # Set status from 'BYTES ONLY / INVALID OR INCOMPLETE'.
                    status = "BYTES ONLY / INVALID OR INCOMPLETE"
                # Handle the remaining case when the preceding conditions did not match.
                else:
                    # Set status from 'LINK SILENT'.
                    status = "LINK SILENT"
                # Print a diagnostic, status, or setup message to the console.
                print("[UART]", status, "bytes=", rx_bytes,
                      # Provide the next argument or expression for print.
                      "valid=", rx_valid, "bad=", rx_bad,
                      # Provide the next argument or expression for print.
                      "last=", last_packet)
            # Yield briefly so the Pico receive loop does not busy-spin.
            sleep_ms(2)
    # Always release resources here, including after an exception or interruption.
    finally:
        # Set the onboard LED output to 0.
        led.value(0)
        # Release the Pico UART hardware during shutdown.
        uart.deinit()
        # Stop the horizontal PWM output during shutdown.
        pan_servo.deinit()
        # Stop the vertical PWM output during shutdown.
        tilt_servo.deinit()


# Continue with this branch only when __name__ == '__main__'.
if __name__ == "__main__":
    # Call main with the arguments below.
    main()
