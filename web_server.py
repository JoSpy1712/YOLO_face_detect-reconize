# Document the purpose of this module, class, or function.
"""Local dashboard sharing the existing recognition process and camera."""
# Compare dashboard passwords using a constant-time comparison helper.
import hmac
# Run background workers and protect state shared between threads.
import threading

# Import the Flask application, response builders, template renderer, and incoming request context.
from flask import Flask, Response, jsonify, render_template, request
# Create the local threaded HTTP server without a debug reloader or a second camera process.
from werkzeug.serving import make_server

# Share status, JPEG frames, and queued requests between camera and browser threads.
from web_state import Monitor
# Translate database failures into messages that do not reveal connection credentials.
from unlock_action import database_error_message



# Build the authenticated Flask dashboard and its video, status, unlock, and history routes.
def create_app(monitor, action, password):
    # Continue with this branch only when not password.
    if not password:
        # Report ValueError('A dashboard password is required').
        raise ValueError("A dashboard password is required")
    # Set app from Flask(__name__).
    app = Flask(__name__)

    # Attach this decorator to the following function or test class.
    @app.before_request
    # Require the admin login and validate the origin and custom header of POST requests.
    def authenticate():
        # Set auth from request.authorization.
        auth = request.authorization
        # Continue with this branch only when not (auth and auth.type == 'basic' and (auth.username == 'admin') and hmac.compare_digest((auth.password or '').encode(), password.encode....
        if not (auth and auth.type == "basic" and auth.username == "admin"
                # Continue the combined condition; all required checks determine this branch.
                and hmac.compare_digest((auth.password or "").encode(), password.encode())):
            # Return Response('Login required', 401, {'WWW-Authenticate': 'Basic realm="Face Monitor"'}) to the caller.
            return Response("Login required", 401,
                            # Continue building the value returned to the caller.
                            {"WWW-Authenticate": 'Basic realm="Face Monitor"'})
        # Continue with this branch only when request.method == 'POST'.
        if request.method == "POST":
            # Browsers cannot send this custom header cross-origin without CORS.
            # Continue with this branch only when request.headers.get('X-Unlock-Request') != '1'.
            if request.headers.get("X-Unlock-Request") != "1":
                # Return (jsonify(error='Invalid request'), 403) to the caller.
                return jsonify(error="Invalid request"), 403
            # Set origin from request.headers.get('Origin').
            origin = request.headers.get("Origin")
            # Continue with this branch only when origin and origin != request.host_url.rstrip('/').
            if origin and origin != request.host_url.rstrip("/"):
                # Return (jsonify(error='Invalid origin'), 403) to the caller.
                return jsonify(error="Invalid origin"), 403

    # Attach this decorator to the following function or test class.
    @app.after_request
    # Apply response headers that disable caching and restrict framing and MIME sniffing.
    def private_response(response):
        # Set response.headers['Cache-Control'] from 'no-store'.
        response.headers["Cache-Control"] = "no-store"
        # Set response.headers['X-Frame-Options'] from 'DENY'.
        response.headers["X-Frame-Options"] = "DENY"
        # Set response.headers['X-Content-Type-Options'] from 'nosniff'.
        response.headers["X-Content-Type-Options"] = "nosniff"
        # Return response to the caller.
        return response

    # Attach this decorator to the following function or test class.
    @app.get("/")
    # Render the HTML template used by the monitoring page.
    def dashboard():
        # Return render_template('monitor.html') to the caller.
        return render_template("monitor.html")

    # Attach this decorator to the following function or test class.
    @app.get("/video_feed")
    # Stream camera JPEGs as an MJPEG multipart HTTP response.
    def video():
        # Return Response(monitor.stream(), mimetype='multipart/x-mixed-replace; boundary=frame') to the caller.
        return Response(monitor.stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # Attach this decorator to the following function or test class.
    @app.get("/api/status")
    # Return a thread-safe JSON snapshot of the application status.
    def status():
        # Return jsonify(monitor.snapshot()) to the caller.
        return jsonify(monitor.snapshot())

    # Attach this decorator to the following function or test class.
    @app.post("/api/unlock")
    # Queue a browser unlock request; the camera loop still rechecks authorization.
    def unlock():
        # Continue with this branch only when not monitor.request_unlock().
        if not monitor.request_unlock():
            # Return (jsonify(error='Unlock unavailable: wait for a fresh owner match or cooldown.'), 409) to the caller.
            return jsonify(error="Unlock unavailable: wait for a fresh owner match or cooldown."), 409
        # Return (jsonify(message='Request queued; check action status for result.'), 202) to the caller.
        return jsonify(message="Request queued; check action status for result."), 202

    # Attach this decorator to the following function or test class.
    @app.get("/api/events")
    # Read capture history and translate database failures into an HTTP error response.
    def events():
        # Attempt this operation and handle the expected failures below.
        try:
            # Return jsonify(events=action.recent_events()) to the caller.
            return jsonify(events=action.recent_events())
        # Handle Exception without losing required cleanup.
        except Exception as error:
            # Call app.logger.exception with the arguments below.
            app.logger.exception("Cannot read capture history")
            # Return (jsonify(error=database_error_message(error)), 503) to the caller.
            return jsonify(error=database_error_message(error)), 503

    # Attach this decorator to the following function or test class.
    @app.get("/api/events/<event_id>/image")
    # Validate the requested MongoDB identifier and retrieve its saved JPEG.
    def event_image(event_id):
        # Attempt this operation and handle the expected failures below.
        try:
            # Set jpeg from action.event_image(event_id).
            jpeg = action.event_image(event_id)
        # Handle ValueError without losing required cleanup.
        except ValueError:
            # Return (jsonify(error='Invalid image ID'), 400) to the caller.
            return jsonify(error="Invalid image ID"), 400
        # Handle Exception without losing required cleanup.
        except Exception:
            # Call app.logger.exception with the arguments below.
            app.logger.exception("Cannot read capture image")
            # Return (jsonify(error='MongoDB image unavailable'), 503) to the caller.
            return jsonify(error="MongoDB image unavailable"), 503
        # Continue with this branch only when jpeg is None.
        if jpeg is None:
            # Return (jsonify(error='Image not found'), 404) to the caller.
            return jsonify(error="Image not found"), 404
        # Return Response(jpeg, mimetype='image/jpeg') to the caller.
        return Response(jpeg, mimetype="image/jpeg")

    # Return app to the caller.
    return app


# Define WebServer to group its related state and methods.
class WebServer:
    # Initialize the object and the resources or state it owns.
    def __init__(self, monitor, action, password, host="0.0.0.0", port=5000):
        # Set self.monitor from monitor.
        self.monitor = monitor
        # Set self.server from make_server(host, port, create_app(monitor, action, password), threaded=True).
        self.server = make_server(host, port, create_app(monitor, action, password), threaded=True)
        # Prepare a background worker without starting it yet.
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    # Start the previously prepared background server thread.
    def start(self):
        # Start self.thread in the background.
        self.thread.start()

    # Shut down this object and release its associated resources.
    def close(self):
        # Call self.monitor.close with the arguments below.
        self.monitor.close()
        # Call self.server.shutdown with the arguments below.
        self.server.shutdown()
        # Call self.server.server_close with the arguments below.
        self.server.server_close()
        # Wait briefly for this worker to finish during shutdown.
        self.thread.join(timeout=2)
