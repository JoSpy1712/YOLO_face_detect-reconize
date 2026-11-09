# Document the purpose of this module, class, or function.
"""Persist an approval capture before submitting an unlock over UART."""
# Create timezone-aware UTC timestamps for stored approval captures.
from datetime import datetime, timezone


# Define CaptureSaveError as a specialized RuntimeError.
class CaptureSaveError(RuntimeError):
    # Document the purpose of this module, class, or function.
    """The approval image was not confirmed saved; UART was not sent."""


# Translate database exceptions into readable messages without revealing credentials.
def database_error_message(error):
    # Document the purpose of this module, class, or function.
    """Explain common database errors without exposing connection credentials."""
    # Import ConfigurationError, ConnectionFailure, OperationFailure from pymongo.errors.
    from pymongo.errors import ConfigurationError, ConnectionFailure, OperationFailure
    # Continue with this branch only when isinstance(error, ConfigurationError).
    if isinstance(error, ConfigurationError):
        # Return 'MongoDB URI or DNS configuration failed. Check MONGODB_URI in .env and any overriding shell variable. For Atlas, verify the cluster is r... to the caller.
        return ("MongoDB URI or DNS configuration failed. Check MONGODB_URI in .env "
                # Continue building the value returned to the caller.
                "and any overriding shell variable. For Atlas, verify the cluster is running "
                # Continue building the value returned to the caller.
                "and copy its current connection string from Connect > Drivers. "
                # Continue building the value returned to the caller.
                "Run .venv/bin/python check_mongodb.py before restarting.")
    # Continue with this branch only when isinstance(error, ConnectionFailure).
    if isinstance(error, ConnectionFailure):
        # Return 'Cannot reach MongoDB. Start the database server or set MONGODB_URI to a running server.' to the caller.
        return "Cannot reach MongoDB. Start the database server or set MONGODB_URI to a running server."
    # Continue with this branch only when isinstance(error, OperationFailure).
    if isinstance(error, OperationFailure):
        # Continue with this branch only when error.code == 18.
        if error.code == 18:
            # Return 'MongoDB login failed. Check the database username, password, and authSource.' to the caller.
            return "MongoDB login failed. Check the database username, password, and authSource."
        # Continue with this branch only when error.code == 13.
        if error.code == 13:
            # Return 'MongoDB access denied. Grant the database user read/write access to the configured database.' to the caller.
            return "MongoDB access denied. Grant the database user read/write access to the configured database."
    # Return 'MongoDB operation failed. Check the terminal for details.' to the caller.
    return "MongoDB operation failed. Check the terminal for details."


# Define UnlockAction to group its related state and methods.
class UnlockAction:
    # Initialize the object and the resources or state it owns.
    def __init__(self, uri, database, collection, uart):
        # Attempt this operation and handle the expected failures below.
        try:
            # Access the Python MongoDB client; this does not install or start a database server.
            from pymongo import MongoClient
        # Handle ImportError without losing required cleanup.
        except ImportError as error:
            # Report RuntimeError('MongoDB requires pymongo: .venv/bin/python -m pip install pymongo').
            raise RuntimeError("MongoDB requires pymongo: .venv/bin/python -m pip install pymongo") from error
        # Import ConfigurationError from pymongo.errors.
        from pymongo.errors import ConfigurationError
        # Attempt this operation and handle the expected failures below.
        try:
            # Set the MongoDB client with bounded network timeouts using MongoClient(uri, serverSelectionTimeoutMS=3000, connectTimeoutMS=3000, socketTimeoutMS=3000).
            self.client = MongoClient(uri, serverSelectionTimeoutMS=3000,
                                      # Supply connectTimeoutMS as a named argument to the enclosing call.
                                      connectTimeoutMS=3000, socketTimeoutMS=3000)
        # Handle ConfigurationError without losing required cleanup.
        except ConfigurationError as error:
            # Report CaptureSaveError(database_error_message(error)).
            raise CaptureSaveError(database_error_message(error)) from None
        # Set the selected MongoDB collection used for approval captures using self.client[database][collection].
        self.collection = self.client[database][collection]
        # Set the shared UART sender, or None when hardware output is disabled using uart.
        self.uart = uart

    # Save the approved JPEG first, then submit the existing APPROVED command over UART.
    def execute(self, jpeg):
        # BSON stores bytes as binary. A 640x480 JPEG fits in one document.
        # Attempt this operation and handle the expected failures below.
        try:
            # Insert the approved capture and retain its database identifier.
            result = self.collection.insert_one({
                # Include the UTC capture timestamp in this record.
                "captured_at": datetime.now(timezone.utc),
                # Include the original JPEG bytes in this record.
                "image": jpeg,
                # Include the JPEG MIME type in this record.
                "content_type": "image/jpeg",
                # Include the authorization label in this record.
                "recognition": "AUTHORIZED",
                # Include the submitted UART command in this record.
                "command": "APPROVED",
                # Include the database-to-UART workflow status in this record.
                "uart_status": "pending" if self.uart else "disabled",
            # Finish the preceding expression, container, or function call.
            })
        # Handle Exception without losing required cleanup.
        except Exception as error:
            # Report CaptureSaveError(database_error_message(error) + ' Unlock command was not sent.').
            raise CaptureSaveError(database_error_message(error) + " Unlock command was not sent.") from error
        # Continue with this branch only when self.uart is None.
        if self.uart is None:
            # Return 'CAPTURE SAVED - UART OFF' to the caller.
            return "CAPTURE SAVED - UART OFF"
        # Attempt this operation and handle the expected failures below.
        try:
            # Send APPROVED only after the database has confirmed the capture insert.
            self.uart.send_command("APPROVED")
        # Handle Exception without losing required cleanup.
        except Exception:
            # Update the stored capture status so the history reflects the UART submission result.
            self.collection.update_one({"_id": result.inserted_id},
                                       # Provide the next argument or expression for self.collection.update_one.
                                       {"$set": {"uart_status": "failed_or_uncertain"}})
            # Report the current exception again.
            raise
        # Attempt this operation and handle the expected failures below.
        try:
            # Update the stored capture status so the history reflects the UART submission result.
            self.collection.update_one({"_id": result.inserted_id},
                                       # Provide the next argument or expression for self.collection.update_one.
                                       {"$set": {"uart_status": "submitted"}})
        # Handle Exception without losing required cleanup.
        except Exception:
            # Return 'UART SENT - DB STATUS UPDATE FAILED' to the caller.
            return "UART SENT - DB STATUS UPDATE FAILED"
        # Return 'CAPTURE SAVED - UART SENT' to the caller.
        return "CAPTURE SAVED - UART SENT"

    # Return metadata for the 20 newest capture records without loading their JPEG data.
    def recent_events(self):
        # Set rows from self.collection.find({}, {'image': 0}).sort('captured_at', -1).limit(20).
        rows = self.collection.find({}, {"image": 0}).sort("captured_at", -1).limit(20)
        # Return [{'id': str(row['_id']), 'captured_at': row['captured_at'].replace(tzinfo=timezone.utc).isoformat(), 'uart_status': row.get('uart_status'... to the caller.
        return [{"id": str(row["_id"]),
                 # Include the UTC capture timestamp in this record.
                 "captured_at": row["captured_at"].replace(tzinfo=timezone.utc).isoformat(),
                 # Include the database-to-UART workflow status in this record.
                 "uart_status": row.get("uart_status", "unknown")}
                # Continue building the value returned to the caller.
                for row in rows]

    # Validate the requested MongoDB identifier and retrieve its saved JPEG.
    def event_image(self, event_id):
        # Import ObjectId from bson.
        from bson import ObjectId
        # Import InvalidId from bson.errors.
        from bson.errors import InvalidId
        # Attempt this operation and handle the expected failures below.
        try:
            # Set identifier from ObjectId(event_id).
            identifier = ObjectId(event_id)
        # Handle InvalidId without losing required cleanup.
        except InvalidId as error:
            # Report ValueError('Invalid event ID').
            raise ValueError("Invalid event ID") from error
        # Set row from self.collection.find_one({'_id': identifier}, {'image': 1}).
        row = self.collection.find_one({"_id": identifier}, {"image": 1})
        # Return None if row is None else row.get('image') to the caller.
        return None if row is None else row.get("image")

    # Shut down this object and release its associated resources.
    def close(self):
        # Release MongoDB connections during cleanup.
        self.client.close()
