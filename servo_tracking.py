# Document the purpose of this module, class, or function.
"""Pi5 face-centering controller, independent of camera/OpenCV libraries."""


# Define ServoTracker to group its related state and methods.
class ServoTracker:
    # Initialize the object and the resources or state it owns.
    def __init__(self, pan_direction=-1, tilt_direction=1, speed=45.0):
        # Start both horizontal and vertical angle commands at the center position, 90 degrees.
        self.pan = self.tilt = 90.0
        # Set the sign that adapts horizontal tracking to the physical mounting using pan_direction.
        self.pan_direction = pan_direction
        # Set the sign that adapts vertical tracking to the physical mounting using tilt_direction.
        self.tilt_direction = tilt_direction
        # Set the maximum correction rate in degrees per second at full image error using speed.
        self.speed = speed
        # Set the detection already processed, preventing duplicate angular steps using None.
        self.last_detection = None
        # Set the previous detection timestamp used to calculate a bounded time step using None.
        self.last_time = None
        # Set the selected face center used to keep target selection consistent using None.
        self.target = None

    # Update target-centered pan and tilt only for a fresh detection, with a dead zone and angle limits.
    def update(self, boxes, width, height, detection, captured_at, now):
        # Continue with this branch only when detection == self.last_detection.
        if detection == self.last_detection:
            # Return the current pan and tilt commands without an additional update.
            return self.pan, self.tilt
        # Set the detection already processed, preventing duplicate angular steps using detection.
        self.last_detection = detection
        # Set the bounded elapsed detection time used for angular integration using 0.05 if self.last_time is None else max(0.0, min(0.1, captured_at - self.last_time)).
        dt = 0.05 if self.last_time is None else max(0.0, min(0.1, captured_at - self.last_time))
        # Set the previous detection timestamp used to calculate a bounded time step using captured_at.
        self.last_time = captured_at
        # Continue with this branch only when not boxes or width <= 0 or height <= 0 or (now - captured_at > 0.5).
        if not boxes or width <= 0 or height <= 0 or now - captured_at > 0.5:
            # Set the selected face center used to keep target selection consistent using None.
            self.target = None
            # Return the current pan and tilt commands without an additional update.
            return self.pan, self.tilt
        # Set the face boxes with positive width and height using [b for b in boxes if b[2] > b[0] and b[3] > b[1]].
        valid = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
        # Continue with this branch only when not valid.
        if not valid:
            # Set the selected face center used to keep target selection consistent using None.
            self.target = None
            # Return the current pan and tilt commands without an additional update.
            return self.pan, self.tilt
        # Continue with this branch only when self.target is None.
        if self.target is None:
            # Acquire the largest visible face when no previous target center is available.
            box = max(valid, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        # Handle the remaining case when the preceding conditions did not match.
        else:
            # Set (x, y) from self.target.
            x, y = self.target
            # Keep the face whose center is nearest the previous target center.
            box = min(valid, key=lambda b: ((b[0]+b[2])/2-x)**2 + ((b[1]+b[3])/2-y)**2)
        # Average each pair of box edges to obtain the selected face center in pixels.
        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        # Set the selected face center used to keep target selection consistent using (cx, cy).
        self.target = (cx, cy)
        # Normalize horizontal and vertical displacement so image center is zero and the edges are about +/-1.
        ex, ey = (cx-width/2)/(width/2), (cy-height/2)/(height/2)
        # Dead zone avoids chatter around the image center.
        # Convert normalized image error to a bounded angular step, or zero inside the dead zone.
        def correction(error):
            # Continue with this branch only when abs(error) <= 0.08.
            if abs(error) <= 0.08:
                # Return 0.0 to the caller.
                return 0.0
            # Return max(-1.0, min(1.0, error)) * self.speed * dt to the caller.
            return max(-1.0, min(1.0, error)) * self.speed * dt
        # Set the controller horizontal angle, clamped to 0 through 180 degrees using max(0.0, min(180.0, self.pan + self.pan_direction * correction(ex))).
        self.pan = max(0.0, min(180.0, self.pan + self.pan_direction*correction(ex)))
        # Set the controller vertical angle, clamped to 0 through 180 degrees using max(0.0, min(180.0, self.tilt + self.tilt_direction * correction(ey))).
        self.tilt = max(0.0, min(180.0, self.tilt + self.tilt_direction*correction(ey)))
        # Return the newly computed pan and tilt angle commands to the UART sending loop.
        return self.pan, self.tilt
