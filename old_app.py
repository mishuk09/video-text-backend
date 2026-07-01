from flask import Flask, request, jsonify
# from flask_cors import cross_origin
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
import jwt
import datetime
import os
from dotenv import load_dotenv
from bson import ObjectId


# Load env vars
load_dotenv()

# app = Flask(__name__)

# CORS(app, supports_credentials=True, resources={r"/*": {
#     "origins": ["http://localhost:5173"],
#     "allow_headers": ["Content-Type", "Authorization"],
#     "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
# }})

app = Flask(__name__)

# ✅ Use this exact configuration
CORS(app, 
     origins=["http://localhost:5173"], 
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
  

# Config
MONGO_URI = os.getenv("MONGO_URI")
JWT_SECRET = os.getenv("JWT_SECRET")
client = MongoClient(MONGO_URI)
db = client["Cluster0"]
users_collection = db["users"]
admins_collection = db["admins"]


# Helper: Create JWT
def create_jwt(user_id):
    payload = {
        "user_id": str(user_id),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

# Helper: Verify JWT
def verify_jwt(token):
    try:
        decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return decoded
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# --- ADD THIS FUNCTION RIGHT HERE ---
def get_active_subcollection():
    existing_batches = [name for name in db.list_collection_names() if name.startswith("users.users_batch_")]
    if not existing_batches:
        return "users.users_batch_1"

    latest_batch_name = sorted(existing_batches, key=lambda x: int(x.split("_")[-1]))[-1]
    latest_batch = db[latest_batch_name]

    if latest_batch.count_documents({}) >= 30:
        new_batch_number = int(latest_batch_name.split("_")[-1]) + 1
        return f"users.users_batch_{new_batch_number}"
    else:
        return latest_batch_name

# Register with subcollections (batches inside "users")
# Register with subcollections (batches inside "users")


@app.route("/api/register", methods=["POST"])
def register():
    data = request.json
    name = data.get("name")
    email = data.get("email").lower()
    password = data.get("password")

    # ✅ Full demographics schema
    demographics = {
        "religion": data.get("religion"),
        "gender": data.get("gender"),
        "age": data.get("age"),
        "place_of_residence": data.get("place_of_residence"),
        "father_occupation": data.get("father_occupation"),
        "mother_occupation": data.get("mother_occupation"),
        "household_monthly_income": data.get("household_monthly_income"),
        "education_level": data.get("education_level"),
        "field_of_study": data.get("field_of_study"),
        "university_college_name": data.get("university_college_name"),
        "attended_government_program": data.get("attended_government_program"),
        "has_entrepreneur_family_or_friends": data.get("has_entrepreneur_family_or_friends"),
        "currently_entrepreneur": data.get("currently_entrepreneur"),
        "prior_entrepreneurship_experience": data.get("prior_entrepreneurship_experience"),
        "considered_inclusive_entrepreneur": data.get("considered_inclusive_entrepreneur"),
    }

    # ✅ Parse behavior data safely
    behavior_data = data.get("behavior_data", {})
    if isinstance(behavior_data, str):
        import json
        try:
            behavior_data = json.loads(behavior_data)
        except:
            behavior_data = {}

    # ✅ Handle trait_scores (must be a list of numbers)
    trait_scores = data.get("trait_scores", [])
    if not isinstance(trait_scores, list):
        return jsonify({"error": "trait_scores must be a list"}), 400

    # ✅ Get main "users" document
    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc:
        users_doc = {"_id": "users", "batches": []}

    # ✅ Check for duplicate email
    for batch in users_doc["batches"]:
        for user in batch["users"]:
            if user["email"] == email:
                return jsonify({"error": "Email already registered"}), 400




    # ✅ Determine which batch to use (create new every 30 users)
    if not users_doc["batches"]:
        batch_name = "users_batch_1"
        users_doc["batches"].append({"batch_name": batch_name, "users": []})
    else:
        last_batch = users_doc["batches"][-1]
        if len(last_batch["users"]) >= 30:
            batch_name = f"users_batch_{len(users_doc['batches']) + 1}"
            users_doc["batches"].append({"batch_name": batch_name, "users": []})
        else:
            batch_name = last_batch["batch_name"]

    # ✅ Add user to the selected batch
    for batch in users_doc["batches"]:
        if batch["batch_name"] == batch_name:
            batch["users"].append({
                "name": name,
                "email": email,
                "password": password,
                "created_at": datetime.datetime.utcnow(),
                "demographics": demographics,
                "behavior_data": behavior_data,
                "trait_scores": trait_scores,  # 👈 New field
            })

    # ✅ Save document back
    users_collection.replace_one({"_id": "users"}, users_doc, upsert=True)

    # ✅ Generate token
    token = create_jwt(str(email))

    return jsonify({
        "token": token,
        "batch": batch_name,
        "user": {
            "name": name,
            "email": email,
            "demographics": demographics,
            "behavior_data": behavior_data,
            "trait_scores": trait_scores
        }
    })



# # Register
# @app.route("/api/register", methods=["POST"])
# def register():
#     data = request.json
#     name = data.get("name")
#     email = data.get("email").lower()
#     password = data.get("password")

#     # demographics
#     demographics = {
#         "religion": data.get("religion"),
#         "gender": data.get("gender"),
#         "age": data.get("age"),
#         "place_of_residence": data.get("place_of_residence"),
#         "father_occupation": data.get("father_occupation"),
#         "mother_occupation": data.get("mother_occupation"),
#         "household_monthly_income": data.get("household_monthly_income"),
#         "education_level": data.get("education_level"),
#         "field_of_study": data.get("field_of_study"),
#         "university_college_name": data.get("university_college_name"),
#         "attended_government_program": data.get("attended_government_program"),
#         "has_entrepreneur_family_or_friends": data.get("has_entrepreneur_family_or_friends"),
#         "currently_entrepreneur": data.get("currently_entrepreneur"),
#         "prior_entrepreneurship_experience": data.get("prior_entrepreneurship_experience"),
#         "considered_inclusive_entrepreneur": data.get("considered_inclusive_entrepreneur"),
#     }

#     # behavior data (force dict)
#     behavior_data = data.get("behavior_data", {})
#     if isinstance(behavior_data, str):
#         import json
#         try:
#             behavior_data = json.loads(behavior_data)
#         except:
#             behavior_data = {}

#     if users_collection.find_one({"email": email}):
#         return jsonify({"error": "Email already registered"}), 400

#     user_id = users_collection.insert_one({
#         "name": name,
#         "email": email,
#         "password": password,
#         "created_at": datetime.datetime.utcnow(),
#         "demographics": demographics,
#         "behavior_data": behavior_data
#     }).inserted_id

#     token = create_jwt(user_id)

#     return jsonify({
#         "token": token,
#         "user": {
#             "id": str(user_id),
#             "name": name,
#             "email": email,
#             "demographics": demographics,
#             "behavior_data": behavior_data
#         }
#     })


@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email").lower()
    password = data.get("password")

    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc:
        return jsonify({"error": "No users found"}), 401

    # Search all batches for the email
    for batch in users_doc.get("batches", []):
        for user in batch.get("users", []):
            if user["email"] == email and user["password"] == password:
                token = create_jwt(email)
                return jsonify({
                    "token": token,
                    "user": {
                        "name": user["name"],
                        "email": user["email"]
                    }
                })

    return jsonify({"error": "Invalid credentials"}), 401



#fetch all users (for testing)
@app.route("/api/users", methods=["GET"])
def get_all_users():
    try:
        users = list(users_collection.find({}))  # fetch all users

        # Remove sensitive fields like password
        for user in users:
            user["id"] = str(user["_id"])
            del user["_id"]
            if "password" in user:
                del user["password"]

        return jsonify({"users": users}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to fetch users: {str(e)}"}), 500




# Delete user by ID
@app.route("/api/users/<user_id>", methods=["DELETE"])
def delete_user(user_id):
    try:
        result = users_collection.delete_one({"_id": ObjectId(user_id)})
        
        if result.deleted_count == 0:
            return jsonify({"success": False, "message": "User not found"}), 404
        
        return jsonify({"success": True, "message": f"User {user_id} deleted successfully"}), 200

    except Exception as e:
        return jsonify({"success": False, "message": f"Error deleting user: {str(e)}"}), 500





# Admin Signup
@app.route("/api/admin/signup", methods=["POST"])
def admin_signup():
    try:
        data = request.json
        first_name = data.get("firstName")
        last_name = data.get("lastName")
        email = data.get("email").lower()
        password = data.get("password")  # ⚠️ stored as plain text

        if not first_name or not last_name or not email or not password:
            return jsonify({"success": False, "message": "All fields are required"}), 400

        # Check if admin already exists
        if admins_collection.find_one({"email": email}):
            return jsonify({"success": False, "message": "Email already registered"}), 400

        # Insert new admin
        admin_id = admins_collection.insert_one({
            "firstName": first_name,
            "lastName": last_name,
            "email": email,
            "password": password,  # ⚠️ plain text
            "role": "admin",
            "created_at": datetime.datetime.utcnow()
        }).inserted_id

        # Create JWT
        token = create_jwt(admin_id)

        return jsonify({
            "success": True,
            "message": "Admin registered successfully",
            "token": token,
            "admin": {
                "id": str(admin_id),
                "firstName": first_name,
                "lastName": last_name,
                "email": email,
                "role": "admin"
            }
        }), 201

    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500



# Admin Sign-In
@app.route("/api/admin/signin", methods=["POST"])
def admin_signin():
    try:
        data = request.json
        email = data.get("email").lower()
        password = data.get("password")

        if not email or not password:
            return jsonify({"success": False, "message": "Email and password required"}), 400

        # Find admin
        admin = admins_collection.find_one({"email": email})
        if not admin or admin.get("password") != password:
            return jsonify({"success": False, "message": "Invalid credentials"}), 401

        # Create JWT
        token = create_jwt(admin["_id"])

        return jsonify({
            "success": True,
            "message": "Login successful",
            "token": token,
            "admin": {
                "id": str(admin["_id"]),
                "firstName": admin.get("firstName"),
                "lastName": admin.get("lastName"),
                "email": admin.get("email"),
                "role": admin.get("role", "admin")
            }
        }), 200

    except Exception as e:
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500




#fetch all admins (for testing)
@app.route("/api/admin", methods=["GET"])
def get_all_admins():
    try:
        admins = list(admins_collection.find({}))  # fetch all admins

        # Remove sensitive fields like password
        for admin in admins:
            admin["id"] = str(admin["_id"])
            del admin["_id"]
            if "password" in admin:
                del admin["password"]

        return jsonify({"admins": admins}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to fetch admins: {str(e)}"}), 500




# Google Login (placeholder)
@app.route("/api/login/google", methods=["POST"])
def google_login():
    data = request.json
    google_email = data.get("email")
    name = data.get("name")

    user = users_collection.find_one({"email": google_email})
    if not user:
        user_id = users_collection.insert_one({
            "name": name,
            "email": google_email,
            "google_account": True,
            "created_at": datetime.datetime.utcnow()
        }).inserted_id
    else:
        user_id = user["_id"]

    token = create_jwt(user_id)
    return jsonify({"token": token, "user": {"id": str(user_id), "name": name, "email": google_email}})

# X Login (placeholder)
@app.route("/api/login/x", methods=["POST"])
def x_login():
    data = request.json
    x_email = data.get("email")
    name = data.get("name")

    user = users_collection.find_one({"email": x_email})
    if not user:
        user_id = users_collection.insert_one({
            "name": name,
            "email": x_email,
            "x_account": True,
            "created_at": datetime.datetime.utcnow()
        }).inserted_id
    else:
        user_id = user["_id"]

    token = create_jwt(user_id)
    return jsonify({"token": token, "user": {"id": str(user_id), "name": name, "email": x_email}})
 

@app.route("/api/profile", methods=["GET"])
def profile():
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return jsonify({"error": "No token"}), 401

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return jsonify({"error": "Invalid auth header format"}), 401

    token = parts[1]
    decoded = verify_jwt(token)
    if not decoded:
        return jsonify({"error": "Invalid or expired token"}), 401

    email = decoded["user_id"]

    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc or "batches" not in users_doc:
        return jsonify({"error": "No users found"}), 404

    found_user = None
    for batch in users_doc["batches"]:
        for user in batch["users"]:
            if user["email"] == email:
                found_user = user
                break
        if found_user:
            break

    if not found_user:
        return jsonify({"error": "User not found"}), 404

    found_user.pop("password", None)
    return jsonify({"user": found_user}), 200



# Update Profile
@app.route("/api/update-profile", methods=["POST"])
def update_profile():
    try:
        data = request.get_json()
        email = data.get("email")

        if not email:
            return jsonify({"success": False, "message": "Email is required"}), 400

        # Do not allow password updates here
        update_data = {k: v for k, v in data.items() if k != "password" and v != ""}

        result = users_collection.update_one(
            {"email": email},
            {"$set": update_data},
            upsert=False
        )

        if result.matched_count == 0:
            return jsonify({"success": False, "message": "User not found"}), 404

        return jsonify({
            "success": True,
            "message": f"Profile updated successfully for {email}",
            "updated_fields": update_data
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error updating profile: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
