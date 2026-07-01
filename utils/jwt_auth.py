from functools import wraps
from flask import request, jsonify, current_app
import jwt

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        if "Authorization" in request.headers:
            token = request.headers["Authorization"].split(" ")[1]

        if not token:
            return jsonify({"error": "Token missing"}), 401

        try:
            data = jwt.decode(
                token,
                current_app.config["SECRET_KEY"],
                algorithms=["HS256"]
            )
            current_user_email = data["email"]
        except Exception:
            return jsonify({"error": "Token invalid or expired"}), 401

        return f(current_user_email, *args, **kwargs)
    return decorated
