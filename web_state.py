# Document the purpose of this module, class, or function.
"""Thread-safe shared state for the local face monitoring dashboard."""
# Exchange commands between threads using synchronized queues.
import queue
# Run background workers and protect state shared between threads.
import threading
# Read monotonic clocks and pace loops without depending on wall-clock changes.
import time


# Define Monitor to group its related state and methods.
class Monitor:
    # Initialize the object and the resources or state it owns.
    def __init__(self):
        # Set the condition variable protecting shared status and waking video viewers using threading.Condition().
        self.condition = threading.Condition()
        # Set the newest encoded JPEG shared with dashboard viewers using None.
        self.jpeg = None
        # Set the shared JPEG sequence counter using 0.
        self.frame_id = 0
        # Set the last status-publish timestamp used to detect an offline producer using 0.0.
        self.updated_at = 0.0
        # Set self.status from {}.
        self.status = {}
        # Set the bounded queue holding at most one pending browser unlock request using queue.Queue(maxsize=1).
        self.requests = queue.Queue(maxsize=1)
        # Set the shutdown flag checked by waiting video streams using False.
        self.closed = False

    # Publish a fresh status and optional JPEG, then wake waiting browser streams.
    def publish(self, status, jpeg=None):
        # Enter self.condition and release or restore it automatically on exit.
        with self.condition:
            # Set self.status from dict(status).
            self.status = dict(status)
            # Set the last status-publish timestamp used to detect an offline producer using time.monotonic().
            self.updated_at = time.monotonic()
            # Continue with this branch only when jpeg is not None.
            if jpeg is not None:
                # Set the newest encoded JPEG shared with dashboard viewers using jpeg.
                self.jpeg = jpeg
                # Update the shared JPEG sequence counter using 1.
                self.frame_id += 1
            # Wake all viewers waiting for a new image or a shutdown notification.
            self.condition.notify_all()

    # Copy the shared status and disable unlocking if the producer is offline or stale.
    def snapshot(self):
        # Enter self.condition and release or restore it automatically on exit.
        with self.condition:
            # Set status from dict(self.status).
            status = dict(self.status)
            # Set stale from self.closed or time.monotonic() - self.updated_at > 1.0.
            stale = self.closed or time.monotonic() - self.updated_at > 1.0
            # Continue with this branch only when stale.
            if stale:
                # Call status.update with the arguments below.
                status.update(status="OFFLINE", can_unlock=False)
            # Return status to the caller.
            return status

    # Queue at most one unlock request while the status currently permits it.
    def request_unlock(self):
        # This is only a request; the camera loop rechecks authorization.
        # Continue with this branch only when not self.snapshot().get('can_unlock', False).
        if not self.snapshot().get("can_unlock", False):
            # Return False to the caller.
            return False
        # Attempt this operation and handle the expected failures below.
        try:
            # Call self.requests.put_nowait with the arguments below.
            self.requests.put_nowait(time.monotonic())
        # Handle queue.Full without losing required cleanup.
        except queue.Full:
            # Return False to the caller.
            return False
        # Return True to the caller.
        return True

    # Consume one queued request and reject requests older than one second.
    def consume_unlock(self):
        # Attempt this operation and handle the expected failures below.
        try:
            # Set requested_at from self.requests.get_nowait().
            requested_at = self.requests.get_nowait()
        # Handle queue.Empty without losing required cleanup.
        except queue.Empty:
            # Return False to the caller.
            return False
        # Return time.monotonic() - requested_at < 1.0 to the caller.
        return time.monotonic() - requested_at < 1.0

    # Wait for a new JPEG and yield correctly framed MJPEG chunks until shutdown.
    def stream(self):
        # Set last_frame from -1.
        last_frame = -1
        # Repeat this loop while True.
        while True:
            # Enter self.condition and release or restore it automatically on exit.
            with self.condition:
                # Call self.condition.wait_for with the arguments below.
                self.condition.wait_for(
                    # Provide the next argument or expression for self.condition.wait_for.
                    lambda: self.closed or (self.jpeg is not None and self.frame_id != last_frame),
                    # Supply timeout as a named argument to the enclosing call.
                    timeout=1,
                # Finish the preceding expression, container, or function call.
                )
                # Continue with this branch only when self.closed.
                if self.closed:
                    # Leave this function without a result.
                    return
                # Continue with this branch only when self.jpeg is None or self.frame_id == last_frame.
                if self.jpeg is None or self.frame_id == last_frame:
                    # Skip the rest of this iteration and wait for the next input or frame.
                    continue
                # Set (jpeg, last_frame) from (self.jpeg, self.frame_id).
                jpeg, last_frame = self.jpeg, self.frame_id
            # Yield the next multipart JPEG chunk to the HTTP video response.
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   # Continue the enclosing expression with this value or condition.
                   + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")

    # Shut down this object and release its associated resources.
    def close(self):
        # Enter self.condition and release or restore it automatically on exit.
        with self.condition:
            # Set the shutdown flag checked by waiting video streams using True.
            self.closed = True
            # Wake all viewers waiting for a new image or a shutdown notification.
            self.condition.notify_all()

