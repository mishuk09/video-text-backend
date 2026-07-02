from flask import Flask, request, jsonify,send_file,send_from_directory
from flask_cors import CORS, cross_origin
import datetime, json
from functools import wraps
import jwt
import os
import uuid
from dotenv import load_dotenv
from bson import ObjectId
import pandas as pd
from flask import send_file
import io
import numpy as np
# from pso import run_pso, load_dataset, comput e_group_score
from pso import (
    run_pso as run_pso_new,
    load_dataset as load_dataset_new,
    compute_group_score_from_values,
    compute_group_score_word,
)
from psov1 import (
    run_pso as run_pso_old,
    load_dataset as load_dataset_old,
    compute_group_score_from_values as compute_group_score_from_values_old,
)
from werkzeug.utils import secure_filename
import time
from routes.post import post_bp
from config.db import db
from utils.jwt_auth import token_required
import re


# -----------------------------------------
# 🔧 CONFIG
# -----------------------------------------
load_dotenv()
app = Flask(__name__)
 

app.register_blueprint(post_bp, url_prefix="/api")

# ✅ Enable CORS for all origins during local development
CORS(
    app,
    resources={r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
    }},
    supports_credentials=False,
)


app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "your_secret_key")


def delete_file_if_exists(file_path):
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            print("Failed to delete file:", e)

#email normalization helper (used in multiple places for consistent matching)
def normalize_email(email):
    if not email:
        return None
    email = email.strip().lower()
    email = re.sub(r"\s+", "", email)  # remove ALL whitespace including tabs
    return email

# MongoDB connection
JWT_SECRET = os.getenv("JWT_SECRET")

users_collection = db["users"]
admins_collection = db["admins"]
connections_collection = db["connections"]
training_resources_collection = db["training_resources"]

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

UPLOAD_ROOT = "uploads"
PROFILE_FOLDER = os.path.join(UPLOAD_ROOT, "profile")
COVER_FOLDER = os.path.join(UPLOAD_ROOT, "cover")

os.makedirs(PROFILE_FOLDER, exist_ok=True)
os.makedirs(COVER_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_FILE_SIZE_MB = 5

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def normalize_email(value):
    return (value or "").strip().lower()


def find_user_snapshot(email):
    """Return sanitized user doc and batch for a given email."""
    normalized_email = normalize_email(email)
    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc:
        return None, None

    for batch in users_doc.get("batches", []):
        for user in batch.get("users", []):
            if user.get("email", "").lower() == normalized_email:
                user_copy = user.copy()
                user_copy.pop("password", None)
                user_copy.pop("confirmPassword", None)
                return user_copy, batch.get("batch_name")

    return None, None


def is_admin_email(email):
    normalized_email = normalize_email(email)
    return admins_collection.find_one({"email": normalized_email}) is not None


# -----------------------------------------
# 🔐 JWT Helper Functions
# -----------------------------------------
def create_jwt(email):
    """Generate JWT token with email"""
    token = jwt.encode(
        {"email": email, "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)},
        app.config["SECRET_KEY"],
        algorithm="HS256"
    )
    return token

 

# -----------------------------------------
# PSO -v1
# -----------------------------------------

@app.route("/api/admin/pso-run", methods=["POST", "OPTIONS"])
@cross_origin()
def run_pso_api():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        file_path = os.path.join(UPLOAD_FOLDER, "dataset.xlsx")
        file.save(file_path)

        df = pd.read_excel(file_path)

        # --- robust: take only survey_* columns, numeric only ---
        survey = df.filter(regex=r"^survey_").copy()
        survey = survey.apply(pd.to_numeric, errors="coerce").fillna(0)

        D  = survey.filter(regex=r"^survey_D").sum(axis=1)
        H  = survey.filter(regex=r"^survey_H\d+").sum(axis=1)     # only H-numbered (H26..H48)
        T  = survey.filter(regex=r"^survey_T").sum(axis=1)

        Hip = survey.filter(regex=r"^survey_Hip").sum(axis=1)
        Hac = survey.filter(regex=r"^survey_Hac").sum(axis=1)
        Hus = survey.filter(regex=r"^survey_Hus").sum(axis=1)

        df_new = pd.DataFrame({"D": D, "H": H, "T": T, "Hip": Hip, "Hus": Hus, "Hac": Hac})


        # Save reduced dataset (like original flow)
        dataset_path = os.path.join(UPLOAD_FOLDER, "dataset.xlsx")
        df_new.to_excel(dataset_path, index=False)

        # Load dataset once via pso.load_dataset (this ensures numeric-only and float32 conversion)
        df_loaded = load_dataset_new(dataset_path)  # uses optimized loader

        # Run PSO with loaded df and word-model scoring enabled
        pos, val, hist, groups = run_pso_new(
            df_loaded, max_iter=100, num_particles=30, n_mem=3, 
            scoring="min", verbose=False, use_word_model=True
        )

        # Compute per-group fit using word-model scoring
        from pso import compute_group_score_word
        fit = [compute_group_score_word(df_loaded, members)[0] for members in groups.values()]

        # Internal best index (0-based, for groups dict)
        if len(fit) > 0:
            best_group_pos = int(np.argmax(fit))   # 0-based internal index
            best_score = float(np.max(fit))
        else:
            best_group_pos = 0
            best_score = float(val)

        # Display group number should start from 1
        best_group_number = best_group_pos + 1

        # -----------------------------
        # Build detailed report (same format as new-pso.py output)
        # -----------------------------
        names = df.iloc[:, 0]
        df_scoring = df_new.copy()

        group_scores = []
        group_leaders = {}
        for g, members in groups.items():
            s, leaders = compute_group_score_word(df_scoring, members)
            group_scores.append(s)
            group_leaders[g] = leaders

        best_group_score = max(group_scores) if group_scores else 0.0
        min_group_score = min(group_scores) if group_scores else 0.0
        avg_group_score = float(np.mean(group_scores)) if group_scores else 0.0
        median_score = float(np.median(group_scores)) if group_scores else 0.0

        output_path = os.path.join(OUTPUT_FOLDER, "group_final.txt")
        components = ["D", "H", "T", "Hip", "Hus", "Hac"]

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write("QALIB GROUP MATCHING RESULTS (WORD-DOCUMENT FORMULA)\n")
            f.write("=" * 100 + "\n")
            f.write(f"Total Participants: {len(df_scoring)}\n")
            f.write(f"Total Groups Formed: {len(groups)}\n")
            f.write(f"Best Group Score (Word model): {best_group_score:.6f}\n")
            f.write(f"Average Group Score (Word model): {avg_group_score:.6f}\n")
            f.write(f"min Score (Word model): {min_group_score:.6f}\n")
            f.write(f"median Group Score (Word model): {median_score:.6f}\n")
            f.write("=" * 100 + "\n\n")

            f.write("GROUP SUMMARY\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Group':<8} | {'Score':<12} | Members\n")
            f.write("-" * 100 + "\n")

            for g, members in groups.items():
                score, _ = compute_group_score_word(df_scoring, members)
                member_names = ", ".join(names.iloc[members].astype(str))
                f.write(f"{g:<8} | {score:<12.6f} | {member_names}\n")

            f.write("\n" + "=" * 100 + "\n")
            f.write("DETAILED GROUP INFORMATION\n")
            f.write("=" * 100 + "\n\n")

            for g, members in groups.items():
                score, leaders = compute_group_score_word(df_scoring, members)

                f.write("=" * 100 + "\n")
                f.write(f"GROUP {g}\n")
                f.write("=" * 100 + "\n")
                f.write(f"Group Score (Word model): {score:.6f}\n")
                f.write(f"Number of Members: {len(members)}\n\n")

                f.write("Component leaders (Qalb: D/H/T different, DreamTeam: Hip/Hus/Hac different):\n")
                for c in components:
                    leader_idx = leaders.get(c)
                    if leader_idx is None:
                        continue
                    leader_name = names.iloc[leader_idx]
                    leader_score = df_scoring.loc[leader_idx, c]
                    f.write(f"  {c}: {leader_name} (score = {leader_score})\n")
                f.write("\n")

                header = "#  | Name                           | " + " | ".join([f"{c:<6}" for c in components])
                f.write(header + "\n")
                f.write("-" * 100 + "\n")

                for idx, m in enumerate(members, start=1):
                    row_vals = [df_scoring.loc[m, c] for c in components]
                    row = f"{idx:<2} | {str(names.iloc[m])[:30]:<30} | " + " | ".join([f"{v:<6}" for v in row_vals])
                    f.write(row + "\n")

                f.write("\n")

        return jsonify({
            "status": "success",
            # Return display group number starting from 1
            "best_group_index": int(best_group_number),
            "best_group": groups[best_group_pos],
            "best_score": float(best_score),
            "download_url": "/api/admin/download/group_final.txt",
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500




# -----------------------------------------
# PSO  v2
# -----------------------------------------

@app.route("/api/admin/pso-run-v2", methods=["POST", "OPTIONS"])
@cross_origin()
def run_pso_api_v2():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        file_path = os.path.join(UPLOAD_FOLDER, "dataset.xlsx")
        file.save(file_path)

        # Read original excel and prepare the same reduced df_new you had before
        df = pd.read_excel(file_path)
        # Keep same slicing as original and force numeric for safe summation
        df_subset = df.iloc[:, 39:140].apply(pd.to_numeric, errors="coerce").fillna(0)

        D = df_subset.iloc[:, 0:25].sum(axis=1)
        H = df_subset.iloc[:, 25:48].sum(axis=1)
        T = df_subset.iloc[:, 48:63].sum(axis=1)
        DT1 = df_subset.iloc[:, 77:82].sum(axis=1)
        DT2 = df_subset.iloc[:, 83:88].sum(axis=1)
        DT3 = df_subset.iloc[:, 89:94].sum(axis=1)

        df_new = pd.DataFrame(
            {"D": D, "H": H, "T": T, "DT1": DT1, "DT2": DT2, "DT3": DT3}
        )

        # Save reduced dataset (like original flow)
        dataset_path = os.path.join(UPLOAD_FOLDER, "dataset.xlsx")
        df_new.to_excel(dataset_path, index=False)

        # Load dataset once via pso.load_dataset (this ensures numeric-only and float32 conversion)
        df_loaded = load_dataset_old(dataset_path)  # uses optimized loader

        # Run PSO with loaded df
        pos, val, hist, groups = run_pso_old(
            df_loaded, max_iter=100, num_particles=30, n_mem=3, scoring="min", verbose=False
        )

        # Compute per-group fit quickly using numpy
        arr_vals = df_loaded.values
        fit = [compute_group_score_from_values_old(arr_vals, members) for members in groups.values()]

        # Internal best index (0-based, for groups dict)
        if len(fit) > 0:
            best_group_pos = int(np.argmax(fit))   # 0-based internal index
            best_score = float(np.max(fit))
        else:
            best_group_pos = 0
            best_score = float(val)

        # Display group number should start from 1
        best_group_number = best_group_pos + 1

        # -----------------------------
        # Build detailed TXT report (same layout style as /api/admin/pso-run)
        # -----------------------------
        name_series = df["fullName"] if "fullName" in df.columns else df.iloc[:, 0]
        df_scoring = df_new.copy()
        components = ["D", "H", "T", "DT1", "DT2", "DT3"]

        group_scores = []
        group_leaders = {}
        for g, members in groups.items():
            s = compute_group_score_from_values_old(arr_vals, members)
            group_scores.append(s)

            leaders = {}
            for c in components:
                leader_idx = df_scoring.loc[members, c].idxmax() if len(members) > 0 else None
                leaders[c] = leader_idx
            group_leaders[g] = leaders

        best_group_score = max(group_scores) if group_scores else 0.0
        min_group_score = min(group_scores) if group_scores else 0.0
        avg_group_score = float(np.mean(group_scores)) if group_scores else 0.0
        median_score = float(np.median(group_scores)) if group_scores else 0.0

        output_path = os.path.join(OUTPUT_FOLDER, "group_final.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write("QALIB GROUP MATCHING RESULTS (PSO V2)\n")
            f.write("=" * 100 + "\n")
            f.write(f"Total Participants: {len(df_scoring)}\n")
            f.write(f"Total Groups Formed: {len(groups)}\n")
            f.write(f"Best Group Score: {best_group_score:.6f}\n")
            f.write(f"Average Group Score: {avg_group_score:.6f}\n")
            f.write(f"min Score: {min_group_score:.6f}\n")
            f.write(f"median Group Score: {median_score:.6f}\n")
            f.write("=" * 100 + "\n\n")

            f.write("GROUP SUMMARY\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Group':<8} | {'Score':<12} | Members\n")
            f.write("-" * 100 + "\n")

            for g, members in groups.items():
                score = compute_group_score_from_values_old(arr_vals, members)
                member_names = ", ".join(name_series.iloc[members].astype(str).tolist())
                f.write(f"{g:<8} | {score:<12.6f} | {member_names}\n")

            f.write("\n" + "=" * 100 + "\n")
            f.write("DETAILED GROUP INFORMATION\n")
            f.write("=" * 100 + "\n\n")

            for g, members in groups.items():
                score = compute_group_score_from_values_old(arr_vals, members)
                leaders = group_leaders.get(g, {})

                f.write("=" * 100 + "\n")
                f.write(f"GROUP {g}\n")
                f.write("=" * 100 + "\n")
                f.write(f"Group Score: {score:.6f}\n")
                f.write(f"Number of Members: {len(members)}\n\n")

                f.write("Component leaders:\n")
                for c in components:
                    leader_idx = leaders.get(c)
                    if leader_idx is None:
                        continue
                    leader_name = name_series.iloc[leader_idx]
                    leader_score = df_scoring.loc[leader_idx, c]
                    f.write(f"  {c}: {leader_name} (score = {leader_score})\n")
                f.write("\n")

                header = "#  | Name                           | " + " | ".join([f"{c:<6}" for c in components])
                f.write(header + "\n")
                f.write("-" * 100 + "\n")

                for idx, m in enumerate(members, start=1):
                    row_vals = [df_scoring.loc[m, c] for c in components]
                    row = f"{idx:<2} | {str(name_series.iloc[m])[:30]:<30} | " + " | ".join([f"{v:<6}" for v in row_vals])
                    f.write(row + "\n")

                f.write("\n")

        return jsonify({
            "status": "success",
            # Return display group number starting from 1
            "best_group_index": int(best_group_number),
            "best_group": groups[best_group_pos],
            "best_score": float(best_score),
            "download_url": "/api/admin/download/group_final.txt",
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500





@app.route("/api/admin/download/<path:filename>", methods=["GET", "OPTIONS"])
@cross_origin()
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)




@app.route("/api/admin/training-resources", methods=["POST", "OPTIONS"])
@cross_origin()
@token_required
def add_training_resource(current_user_email):
    current_user_email = normalize_email(current_user_email)

    if not is_admin_email(current_user_email):
        return jsonify({"error": "Admin access required"}), 403

    data = request.get_json() or {}
    title = (data.get("title") or "").strip()
    video_url = (data.get("videoUrl") or data.get("video_url") or data.get("url") or "").strip()
    resource_type = (data.get("type") or "").strip().lower()

    # Validate required fields
    if not title or not video_url:
        return jsonify({"error": "title and videoUrl are required"}), 400
    
    # Validate type field
    if not resource_type or resource_type not in ["video", "ppt"]:
        return jsonify({"error": "type must be either 'video' or 'ppt'"}), 400

    resource = {
        "title": title,
        "videoUrl": video_url,
        "type": resource_type,
        "createdBy": current_user_email,
        "created_at": datetime.datetime.utcnow(),
    }

    description = (data.get("description") or "").strip()
    if description:
        resource["description"] = description

    result = training_resources_collection.insert_one(resource)

    return jsonify({
        "message": "Training resource added successfully",
        "resource": {
            "id": str(result.inserted_id),
            "title": resource["title"],
            "videoUrl": resource["videoUrl"],
            "type": resource["type"],
            "description": resource.get("description"),
            "createdBy": resource["createdBy"],
            "created_at": resource["created_at"].isoformat() + "Z",
        }
    }), 201

 
@app.route("/api/training-resources", methods=["GET"])
def get_training_resources():
    resources = list(training_resources_collection.find().sort("created_at", -1))

    serialized_resources = []
    for resource in resources:
        serialized_resources.append({
            "id": str(resource.get("_id")),
            "title": resource.get("title"),
            "videoUrl": resource.get("videoUrl"),
            "type": resource.get("type", "video"),
            "description": resource.get("description"),
            "createdBy": resource.get("createdBy"),
            "created_at": resource.get("created_at").isoformat() + "Z" if resource.get("created_at") else None,
        })

    return jsonify({"resources": serialized_resources, "count": len(serialized_resources)}), 200


@app.route("/api/admin/training-resources/<resource_id>", methods=["PUT", "OPTIONS"])
@cross_origin()
@token_required
def edit_training_resource(current_user_email, resource_id):
    current_user_email = normalize_email(current_user_email)

    if not is_admin_email(current_user_email):
        return jsonify({"error": "Admin access required"}), 403

    # Validate ObjectId
    try:
        object_id = ObjectId(resource_id)
    except Exception:
        return jsonify({"error": "Invalid resource ID"}), 400

    data = request.get_json() or {}
    
    # Prepare update data
    update_data = {}
    
    title = (data.get("title") or "").strip()
    if title:
        update_data["title"] = title
    
    video_url = (data.get("videoUrl") or data.get("video_url") or data.get("url") or "").strip()
    if video_url:
        update_data["videoUrl"] = video_url
    
    resource_type = (data.get("type") or "").strip().lower()
    if resource_type:
        # Validate type field
        if resource_type not in ["video", "ppt"]:
            return jsonify({"error": "type must be either 'video' or 'ppt'"}), 400
        update_data["type"] = resource_type
    
    description = (data.get("description") or "").strip()
    if "description" in data:  # Allow clearing description
        update_data["description"] = description if description else None
    
    # Add updated_at timestamp
    update_data["updated_at"] = datetime.datetime.utcnow()

    if not update_data or len(update_data) == 1:  # Only has updated_at
        return jsonify({"error": "No fields to update"}), 400

    # Update the resource
    result = training_resources_collection.update_one(
        {"_id": object_id},
        {"$set": update_data}
    )

    if result.matched_count == 0:
        return jsonify({"error": "Training resource not found"}), 404

    # Fetch and return updated resource
    updated_resource = training_resources_collection.find_one({"_id": object_id})
    
    return jsonify({
        "message": "Training resource updated successfully",
        "resource": {
            "id": str(updated_resource.get("_id")),
            "title": updated_resource.get("title"),
            "videoUrl": updated_resource.get("videoUrl"),
            "type": updated_resource.get("type", "video"),
            "description": updated_resource.get("description"),
            "createdBy": updated_resource.get("createdBy"),
            "created_at": updated_resource.get("created_at").isoformat() + "Z",
            "updated_at": updated_resource.get("updated_at").isoformat() + "Z" if updated_resource.get("updated_at") else None,
        }
    }), 200


@app.route("/api/admin/training-resources/<resource_id>", methods=["DELETE", "OPTIONS"])
@cross_origin()
@token_required
def delete_training_resource(current_user_email, resource_id):
    current_user_email = normalize_email(current_user_email)

    if not is_admin_email(current_user_email):
        return jsonify({"error": "Admin access required"}), 403

    # Validate ObjectId
    try:
        object_id = ObjectId(resource_id)
    except Exception:
        return jsonify({"error": "Invalid resource ID"}), 400

    # Delete the resource
    result = training_resources_collection.delete_one({"_id": object_id})

    if result.deleted_count == 0:
        return jsonify({"error": "Training resource not found"}), 404

    return jsonify({
        "message": "Training resource deleted successfully",
        "id": resource_id
    }), 200






# Updated Backend Endpoints for Profile and Cover Photo Upload
# Replace your old endpoints with these updated versions
 
@app.route("/api/user/upload-profile-photo", methods=["POST"])
@cross_origin()
@token_required
def upload_profile_photo(current_user_email):
    
    data = request.get_json()
    if not data or "profilePhotoUrl" not in data:
        return jsonify({"error": "No photo URL provided"}), 400
    
    profile_photo_url = data["profilePhotoUrl"]
    
    # Normalize email for matching
    current_user_email = (current_user_email or "").lower()
    print(f"🔍 Uploading profile photo for: {current_user_email}")

    # ✅ Use MongoDB array filters (same as add-survey)
    result = users_collection.update_one(
        {"_id": "users"},
        {"$set": {
            "batches.$[batch].users.$[user].profilePhoto": {
                "url": profile_photo_url,
                "path": profile_photo_url
            }
        }},
        array_filters=[
            {"batch.users": {"$exists": True}},
            {"user.email": current_user_email}
        ]
    )
    
    print(f"💾 Database update - Modified: {result.modified_count}")
    
    if result.modified_count == 0:
        print(f"❌ User not found for: {current_user_email}")
        return jsonify({"error": "User not found"}), 404
    
    return jsonify({
        "message": "Profile photo uploaded successfully",
        "profilePhotoUrl": profile_photo_url,
        "userEmail": current_user_email
    }), 200


@app.route("/api/user/upload-cover-photo", methods=["POST"])
@cross_origin()
@token_required
def upload_cover_photo(current_user_email):
    """
    Upload cover photo to Cloudinary and store URL in database
    """
    data = request.get_json()
    
    if not data or "coverPhotoUrl" not in data:
        return jsonify({"error": "No photo URL provided"}), 400
    
    cover_photo_url = data["coverPhotoUrl"]
    
    # Normalize email for matching
    current_user_email = (current_user_email or "").lower()
    print(f"🔍 Uploading cover photo for: {current_user_email}")

    # ✅ Use MongoDB array filters (same as add-survey)
    result = users_collection.update_one(
        {"_id": "users"},
        {"$set": {
            "batches.$[batch].users.$[user].coverPhoto": {
                "url": cover_photo_url,
                "path": cover_photo_url
            }
        }},
        array_filters=[
            {"batch.users": {"$exists": True}},
            {"user.email": current_user_email}
        ]
    )
    
    print(f"💾 Database update - Modified: {result.modified_count}")
    
    if result.modified_count == 0:
        print(f"❌ User not found for: {current_user_email}")
        return jsonify({"error": "User not found"}), 404
    
    return jsonify({
        "message": "Cover photo uploaded successfully",
        "coverPhotoUrl": cover_photo_url,
        "userEmail": current_user_email
    }), 200


# user search functionality
@app.route("/api/search-users", methods=["GET"])
def search_users():
    query = request.args.get("query", "").strip()

    if not query:
        return jsonify({"users": []}), 200

    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc or not users_doc.get("batches"):
        return jsonify({"users": []}), 200

    query_lower = query.lower()
    matched_users = []

    for batch in users_doc.get("batches", []):
        for user in batch.get("users", []):
            full_name = user.get("fullName", "").lower()

            if query_lower in full_name:
                user_copy = user.copy()

                # remove sensitive fields
                user_copy.pop("password", None)
                user_copy.pop("confirmPassword", None)

                user_copy["batch_name"] = batch.get("batch_name")
                matched_users.append(user_copy)

    return jsonify({
        "query": query,
        "count": len(matched_users),
        "users": matched_users
    }), 200



# -----------------------------------------
# 🧠 UPDATE survey
# -----------------------------------------
@app.route("/api/update-survey", methods=["POST"])
@token_required
def update_profile(current_user_email):
    """
    Update parts of the user (survey, dreamteam, bigfive, etc.)
    """
    data = request.json

    # Identify which section to update (dynamic hybrid)
    update_data = {}
    allowed_sections = ["survey", "dreamteam", "bigfive", "cohortinformation", "demographics"]

    for section in allowed_sections:
        if section in data:
            update_data[f"batches.$[batch].users.$[user].{section}"] = data[section]

    if not update_data:
        return jsonify({"error": "No valid section to update"}), 400

    # ✅ Perform the update in nested structure
    result = users_collection.update_one(
        {"_id": "users"},
        {"$set": update_data},
        array_filters=[
            {"batch.users": {"$exists": True}},
            {"user.email": current_user_email}
        ]
    )

    if result.modified_count == 0:
        return jsonify({"error": "User not found"}), 404

    return jsonify({"message": "Profile updated successfully"}), 200


 


# -----------------------------------------
# 🧠 ADD SURVEY
# -----------------------------------------



@app.route("/api/add-survey", methods=["POST"])
@token_required
def add_survey(current_user_email):
    """
    Update parts of the user (add survey, dreamteam, bigfive, etc.)
    When survey is submitted, issurveyDone will be set to true
    """
    data = request.json

    # Identify which section to update (dynamic hybrid)
    update_data = {}
    allowed_sections = ["survey", "dreamteam", "bigfive", "cohortinformation", "demographics"]

    for section in allowed_sections:
        if section in data:
            update_data[f"batches.$[batch].users.$[user].{section}"] = data[section]

    # ✅ If survey is being submitted, mark issurveyDone as true
    if "survey" in data:
        update_data["batches.$[batch].users.$[user].issurveyDone"] = True

    if not update_data:
        return jsonify({"error": "No valid section to update"}), 400

    # ✅ Perform the update in nested structure
    result = users_collection.update_one(
        {"_id": "users"},
        {"$set": update_data},
        array_filters=[
            {"batch.users": {"$exists": True}},
            {"user.email": current_user_email}
        ]
    )

    if result.modified_count == 0:
        return jsonify({"error": "User not found"}), 404

    return jsonify({"message": "Profile updated successfully"}), 200


# 🧠 ADD BIG FIVE DATA
@app.route("/api/add-bigfive", methods=["POST"])
@token_required
def add_bigfive(current_user_email):
    """
    Add/Update Big Five personality test data
    Expects: {bigfive: {q1, q2, q3, q4, q5, q6, q7, q8, q9, q10}}
    Same pattern as add-survey
    """
    data = request.json
    
    # Identify which section to update (bigfive only)
    update_data = {}
    
    if "bigfive" in data:
        update_data["batches.$[batch].users.$[user].bigfive"] = data["bigfive"]
    
    else:
        return jsonify({"error": "No bigfive data provided"}), 400
    
    # ✅ If bigfive is being submitted, mark isBigFiveDone as true
    if "bigfive" in data:
        update_data["batches.$[batch].users.$[user].isBigFiveDone"] = True

    if not update_data:
        return jsonify({"error": "No valid data to update"}), 400

    # ✅ Perform the update in nested structure
    result = users_collection.update_one(
        {"_id": "users"},
        {"$set": update_data},
        array_filters=[
            {"batch.users": {"$exists": True}},
            {"user.email": current_user_email}
        ]
    )

    if result.modified_count == 0:
        return jsonify({"error": "User not found"}), 404

    return jsonify({
        "message": "Big Five data saved successfully",
        "bigfive": data["bigfive"]
    }), 200


# -----------------------------------------
# 🧩 REGISTER ENDPOINT
# -----------------------------------------
@app.route("/api/register", methods=["POST"])
def register():
    data = request.json
    full_name = (data.get("fullName") or "").strip()
    email = normalize_email(data.get("email"))
    password = data.get("password") or ""
    confirm_password = data.get("confirmPassword") or ""
    role = (data.get("role") or "user").strip().lower() or "user"

    if not full_name or not email or not password or not confirm_password:
        return jsonify({"error": "fullName, email, password and confirmPassword are required"}), 400

    if password != confirm_password:
        return jsonify({"error": "Password and confirmPassword do not match"}), 400

    existing_flat_user = users_collection.find_one({"email": email})
    if existing_flat_user:
        return jsonify({"error": "Email already registered"}), 400

    legacy_user, _ = find_user_snapshot(email)
    if legacy_user:
        return jsonify({"error": "Email already registered"}), 400

    new_user = {
        "fullName": full_name,
        "email": email,
        "password": password,
        "confirmPassword": confirm_password,
        "role": role,
        "created_at": datetime.datetime.utcnow(),
    }

    users_collection.insert_one(new_user)

    token = create_jwt(email)
    response_user = new_user.copy()
    response_user["created_at"] = response_user["created_at"].isoformat() + "Z"

    return jsonify({
        "token": token,
        "user": response_user
    }), 201

 
# -----------------------------------------
# 🔑 SIGNIN ENDPOINT
# -----------------------------------------
@app.route("/api/signin", methods=["POST"])
def signin():
    """User login using email and password"""
    data = request.get_json(silent=True) or {}
    email = normalize_email(data.get("email"))
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    found_user = users_collection.find_one({"email": email, "password": password})

    if not found_user:
        return jsonify({"error": "Invalid email or password"}), 401

    # ✅ Generate JWT Token
    token = create_jwt(email)

    return jsonify({
        "message": "Signin successful",
        "token": token,
        "user": {
            "fullName": found_user.get("fullName"),
            "email": found_user.get("email"),
            "role": found_user.get("role", "user"),
            "created_at": found_user.get("created_at")
        }
    }), 200


# -----------------------------------------
# 🔁 RESET PASSWORD ENDPOINT
# -----------------------------------------
@app.route("/api/reset-password", methods=["POST"])
def reset_password():
    """
    Reset password using email + newPassword + confirmPassword.
    """
    data = request.get_json() or {}

    email = normalize_email(data.get("email"))
    new_password = data.get("newPassword", "")
    confirm_password = data.get("confirmPassword", "")

    if not email or not new_password or not confirm_password:
        return jsonify({"error": "Email, newPassword and confirmPassword are required"}), 400

    if new_password != confirm_password:
        return jsonify({"error": "New password and confirm password do not match"}), 400

    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc or not users_doc.get("batches"):
        return jsonify({"error": "No users found"}), 404

    user_found = False

    for batch in users_doc.get("batches", []):
        for user in batch.get("users", []):
            if normalize_email(user.get("email")) == email:
                user["password"] = new_password
                user["confirmPassword"] = confirm_password
                user_found = True
                break
        if user_found:
            break

    if not user_found:
        return jsonify({"error": "User not found"}), 404

    users_collection.replace_one({"_id": "users"}, users_doc, upsert=True)

    return jsonify({"message": "Password reset successful"}), 200



# -----------------------------------------
# 👤 USER PROFILE ENDPOINT
# -----------------------------------------
@app.route("/api/user-profile", methods=["GET"])
@token_required
def user_profile(current_user_email):
    """
    Fetch the user profile based on the JWT token.
    """
    # Get the users document
    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc:
        return jsonify({"error": "No users found"}), 404

    # Find the user in batches
    for batch in users_doc["batches"]:
        for user in batch["users"]:
            if user["email"] == current_user_email:
                # Return user data except password for security
                user_data = user.copy()
                user_data.pop("password", None)
                user_data.pop("confirmPassword", None)
                return jsonify({"user": user_data, "batch": batch["batch_name"]}), 200

    return jsonify({"error": "User not found"}), 404

# -----------------------------------------
# 👤 USER SURVEY ENDPOINT
# -----------------------------------------
@app.route("/api/user-survey", methods=["GET"])
@token_required
def user_survey(current_user_email):
    """
    Fetch the logged-in user's survey data (D1, D2..., H1, H2...) based on JWT token.
    """
    # Get the users document
    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc:
        return jsonify({"error": "No users found"}), 404

    # Find the user in batches
    for batch in users_doc.get("batches", []):
        for user in batch.get("users", []):
            if user.get("email") == current_user_email:
                survey_data = user.get("survey", {})  # only survey data
                if not survey_data:
                    return jsonify({"error": "Survey data not found"}), 404

                return jsonify({
                    "survey": survey_data,
                    "batch": batch.get("batch_name")
                }), 200

    return jsonify({"error": "User not found"}), 404


# -----------------------------------------
# 🤝 USER CONNECTIONS
# -----------------------------------------
@app.route("/api/user/connect", methods=["POST"])
@token_required
def connect_user(current_user_email):
    """Persist a connection initiated by the authenticated user."""
    data = request.get_json() or {}
    target_email = normalize_email(data.get("targetEmail"))
    note = data.get("note")

    current_user_email = normalize_email(current_user_email)

    if not target_email:
        return jsonify({"error": "targetEmail is required"}), 400

    if target_email == current_user_email:
        return jsonify({"error": "You cannot connect with yourself"}), 400

    target_user, batch_name = find_user_snapshot(target_email)
    if not target_user:
        return jsonify({"error": "Target user not found"}), 404

    existing = connections_collection.find_one(
        {"userEmail": current_user_email, "connections.email": target_email},
        {"_id": 1}
    )
    if existing:
        return jsonify({"message": "Connection already exists"}), 200

    connected_at = datetime.datetime.utcnow().isoformat() + "Z"
    connection_payload = {
        "email": target_email,
        "fullName": target_user.get("fullName"),
        "batch": batch_name,
        "profilePhoto": target_user.get("profilePhoto"),
        "cohortinformation": target_user.get("cohortinformation"),
        "demographics": target_user.get("demographics"),
        "connectedAt": connected_at,
    }

    if note:
        connection_payload["note"] = note

    connections_collection.update_one(
        {"userEmail": current_user_email},
        {
            "$setOnInsert": {"userEmail": current_user_email},
            "$push": {"connections": connection_payload}
        },
        upsert=True
    )

    return jsonify({"message": "Connection stored", "connection": connection_payload}), 201


@app.route("/api/user/connections", methods=["GET"])
@token_required
def list_connections(current_user_email):
    """Return all saved connections for the authenticated user."""
    current_user_email = normalize_email(current_user_email)
    doc = connections_collection.find_one({"userEmail": current_user_email})
    connections = doc.get("connections", []) if doc else []

    return jsonify({
        "count": len(connections),
        "connections": connections
    }), 200


@app.route("/api/user/connections", methods=["DELETE"])
@token_required
def delete_connection(current_user_email):
    """Remove a specific connection for the authenticated user."""
    data = request.get_json() or {}
    target_email = normalize_email(data.get("targetEmail"))
    current_user_email = normalize_email(current_user_email)

    if not target_email:
        return jsonify({"error": "targetEmail is required"}), 400

    result = connections_collection.update_one(
        {"userEmail": current_user_email},
        {"$pull": {"connections": {"email": target_email}}}
    )

    if result.modified_count == 0:
        return jsonify({"error": "Connection not found"}), 404

    return jsonify({"message": "Connection removed", "targetEmail": target_email}), 200


# -----------------------------------------
# 📰 USER NEWS
# -----------------------------------------
@app.route("/api/user/news", methods=["POST"])
@token_required
def add_user_news(current_user_email):
    """Store a news item with main title, thumbnail title, description, and tags."""
    data = request.get_json(silent=True) or {}

    target_email = normalize_email(data.get("email"))
    main_title = (
        data.get("mainTitle")
        or data.get("maintitle")
        or data.get("main_title")
        or data.get("title")
        or ""
    ).strip()
    thumbnail_title = (
        data.get("thumbnailTitle")
        or data.get("thumbailtitle")
        or data.get("thumbnailtitle")
        or data.get("thumbnail_title")
        or ""
    ).strip()
    description = (data.get("description") or "").strip()
    raw_tags = data.get("tags", [])

    if isinstance(raw_tags, str):
        tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]
    elif isinstance(raw_tags, (list, tuple)):
        tags = [str(tag).strip() for tag in raw_tags if str(tag).strip()]
    else:
        tags = []

    if not target_email or not main_title or not thumbnail_title or not description:
        return jsonify({
            "error": "email, mainTitle, thumbnailTitle and description are required"
        }), 400

    news_item = {
        "mainTitle": main_title,
        "thumbnailTitle": thumbnail_title,
        "description": description,
        "tags": tags,
        "created_at": datetime.datetime.utcnow()
    }

    # 1) Flat user document style
    flat_result = users_collection.update_one(
        {"email": target_email},
        {"$push": {"news": news_item}}
    )
    if flat_result.modified_count > 0:
        response_news = news_item.copy()
        response_news["created_at"] = response_news["created_at"].isoformat() + "Z"
        return jsonify({
            "message": "News saved successfully",
            "email": target_email,
            "news": response_news
        }), 201

    # 2) Legacy batch-based users document style
    legacy_result = users_collection.update_one(
        {"_id": "users"},
        {"$push": {"batches.$[batch].users.$[user].news": news_item}},
        array_filters=[
            {"batch.users": {"$exists": True}},
            {"user.email": target_email}
        ]
    )

    if legacy_result.modified_count == 0:
        return jsonify({"error": "User not found"}), 404

    response_news = news_item.copy()
    response_news["created_at"] = response_news["created_at"].isoformat() + "Z"
    return jsonify({
        "message": "News saved successfully",
        "email": target_email,
        "news": response_news
    }), 201


@app.route("/api/user/news", methods=["GET"])
@token_required
def get_user_news(current_user_email):
    """Fetch news items for the authenticated user."""
    current_user_email = normalize_email(current_user_email)

    # 1) Flat user document style
    flat_user = users_collection.find_one({"email": current_user_email}) or {}
    flat_news = flat_user.get("news", []) if flat_user else []

    if flat_news:
        serialized_news = []
        for item in flat_news:
            item_copy = item.copy() if isinstance(item, dict) else {"value": item}
            created_at = item_copy.get("created_at")
            if created_at and hasattr(created_at, "isoformat"):
                item_copy["created_at"] = created_at.isoformat() + "Z"
            serialized_news.append(item_copy)

        return jsonify({
            "email": current_user_email,
            "count": len(serialized_news),
            "news": serialized_news,
        }), 200

    # 2) Legacy batch-based users document style
    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc or not users_doc.get("batches"):
        return jsonify({"email": current_user_email, "count": 0, "news": []}), 200

    for batch in users_doc.get("batches", []):
        for user in batch.get("users", []):
            if normalize_email(user.get("email")) == current_user_email:
                news_items = user.get("news", []) or []
                serialized_news = []

                for item in news_items:
                    item_copy = item.copy() if isinstance(item, dict) else {"value": item}
                    created_at = item_copy.get("created_at")
                    if created_at and hasattr(created_at, "isoformat"):
                        item_copy["created_at"] = created_at.isoformat() + "Z"
                    serialized_news.append(item_copy)

                return jsonify({
                    "email": current_user_email,
                    "batch": batch.get("batch_name"),
                    "count": len(serialized_news),
                    "news": serialized_news,
                }), 200

    return jsonify({"email": current_user_email, "count": 0, "news": []}), 200


# -----------------------------------------
# 📝 USER NOTES
# -----------------------------------------
@app.route("/api/user/notes", methods=["POST"])
@token_required
def add_user_note(current_user_email):
    """Store a note with title and description for the authenticated user."""
    data = request.get_json(silent=True) or {}

    current_user_email = normalize_email(current_user_email)
    title = (data.get("title") or "").strip()
    description = (data.get("description") or "").strip()

    if not title or not description:
        return jsonify({"error": "title and description are required"}), 400

    note_item = {
        "noteId": uuid.uuid4().hex,
        "title": title,
        "description": description,
        "created_at": datetime.datetime.utcnow()
    }

    # 1) Flat user document style
    flat_result = users_collection.update_one(
        {"email": current_user_email},
        {"$push": {"notes": note_item}}
    )
    if flat_result.modified_count > 0:
        response_note = note_item.copy()
        response_note["created_at"] = response_note["created_at"].isoformat() + "Z"
        return jsonify({
            "message": "Note saved successfully",
            "email": current_user_email,
            "note": response_note
        }), 201

    # 2) Legacy batch-based users document style
    legacy_result = users_collection.update_one(
        {"_id": "users"},
        {"$push": {"batches.$[batch].users.$[user].notes": note_item}},
        array_filters=[
            {"batch.users": {"$exists": True}},
            {"user.email": current_user_email}
        ]
    )

    if legacy_result.modified_count == 0:
        return jsonify({"error": "User not found"}), 404

    response_note = note_item.copy()
    response_note["created_at"] = response_note["created_at"].isoformat() + "Z"
    return jsonify({
        "message": "Note saved successfully",
        "email": current_user_email,
        "note": response_note
    }), 201


@app.route("/api/user/notes", methods=["GET"])
@token_required
def get_user_notes(current_user_email):
    """Fetch notes for the authenticated user."""
    current_user_email = normalize_email(current_user_email)

    flat_user = users_collection.find_one({"email": current_user_email}) or {}
    flat_notes = flat_user.get("notes", []) if flat_user else []

    if flat_notes:
        serialized_notes = []
        for item in flat_notes:
            item_copy = item.copy() if isinstance(item, dict) else {"value": item}
            created_at = item_copy.get("created_at")
            if created_at and hasattr(created_at, "isoformat"):
                item_copy["created_at"] = created_at.isoformat() + "Z"
            serialized_notes.append(item_copy)

        return jsonify({
            "email": current_user_email,
            "count": len(serialized_notes),
            "notes": serialized_notes,
        }), 200

    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc or not users_doc.get("batches"):
        return jsonify({"email": current_user_email, "count": 0, "notes": []}), 200

    for batch in users_doc.get("batches", []):
        for user in batch.get("users", []):
            if normalize_email(user.get("email")) == current_user_email:
                notes = user.get("notes", []) or []
                serialized_notes = []

                for item in notes:
                    item_copy = item.copy() if isinstance(item, dict) else {"value": item}
                    created_at = item_copy.get("created_at")
                    if created_at and hasattr(created_at, "isoformat"):
                        item_copy["created_at"] = created_at.isoformat() + "Z"
                    serialized_notes.append(item_copy)

                return jsonify({
                    "email": current_user_email,
                    "batch": batch.get("batch_name"),
                    "count": len(serialized_notes),
                    "notes": serialized_notes,
                }), 200

    return jsonify({"email": current_user_email, "count": 0, "notes": []}), 200


@app.route("/api/user/notes/<note_id>", methods=["PUT"])
@token_required
def update_user_note(current_user_email, note_id):
    """Update a specific note for the authenticated user."""
    data = request.get_json(silent=True) or {}
    current_user_email = normalize_email(current_user_email)

    update_data = {}
    if "title" in data:
        title = (data.get("title") or "").strip()
        if not title:
            return jsonify({"error": "title cannot be empty"}), 400
        update_data["title"] = title

    if "description" in data:
        description = (data.get("description") or "").strip()
        if not description:
            return jsonify({"error": "description cannot be empty"}), 400
        update_data["description"] = description

    if not update_data:
        return jsonify({"error": "No fields to update"}), 400

    flat_result = users_collection.update_one(
        {"email": current_user_email, "notes.noteId": note_id},
        {"$set": {f"notes.$.{key}": value for key, value in update_data.items()}}
    )
    if flat_result.modified_count > 0:
        return jsonify({"message": "Note updated successfully", "noteId": note_id}), 200

    legacy_result = users_collection.update_one(
        {"_id": "users", "batches.users.email": current_user_email, "batches.users.notes.noteId": note_id},
        {"$set": {f"batches.$[batch].users.$[user].notes.$[note].{key}": value for key, value in update_data.items()}},
        array_filters=[
            {"batch.users": {"$exists": True}},
            {"user.email": current_user_email},
            {"note.noteId": note_id}
        ]
    )

    if legacy_result.modified_count == 0:
        return jsonify({"error": "Note not found"}), 404

    return jsonify({"message": "Note updated successfully", "noteId": note_id}), 200


@app.route("/api/user/notes/<note_id>", methods=["DELETE"])
@token_required
def delete_user_note(current_user_email, note_id):
    """Delete a specific note for the authenticated user."""
    current_user_email = normalize_email(current_user_email)

    flat_result = users_collection.update_one(
        {"email": current_user_email},
        {"$pull": {"notes": {"noteId": note_id}}}
    )
    if flat_result.modified_count > 0:
        return jsonify({"message": "Note deleted successfully", "noteId": note_id}), 200

    legacy_result = users_collection.update_one(
        {"_id": "users"},
        {"$pull": {"batches.$[batch].users.$[user].notes": {"noteId": note_id}}},
        array_filters=[
            {"batch.users": {"$exists": True}},
            {"user.email": current_user_email}
        ]
    )

    if legacy_result.modified_count == 0:
        return jsonify({"error": "Note not found"}), 404

    return jsonify({"message": "Note deleted successfully", "noteId": note_id}), 200


# -----------------------------------------
# ADMIN
# -----------------------------------------


# -----------------------------------------
# 👥 FETCH ALL USERS
# -----------------------------------------
@app.route("/api/users", methods=["GET"])
def get_all_users():
    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc or not users_doc.get("batches"):
        return jsonify({"users": []}), 200

    all_users = []
    for batch in users_doc["batches"]:
        all_users.extend(batch.get("users", []))

    return jsonify({"users": all_users}), 200



@app.route("/api/users/by-email", methods=["GET"])
def get_user_by_email():
    email = normalize_email(request.args.get("email"))

    if not email:
        return jsonify({"error": "email query parameter is required"}), 400

    user, batch_name = find_user_snapshot(email)
    if not user:
        return jsonify({"error": "User not found"}), 404

    user["batch_name"] = batch_name
    return jsonify({"user": user}), 200



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

# -----------------------------
# 🔑 ADMIN SIGNIN
# -----------------------------
@app.route("/api/admin/signin", methods=["POST"])
def admin_signin():
    data = request.json
    email = data.get("email", "").lower()
    password = data.get("password", "")

    admins_collection = db["admins"]
    admin = admins_collection.find_one({"email": email, "password": password})

    if not admin:
        return jsonify({"error": "Invalid email or password"}), 401

    # Create token for admin (optional: can use separate secret if needed)
    token = create_jwt(email)
    return jsonify({"token": token, "admin": {"email": admin["email"], "name": admin.get("name", "")}}), 200

#-----------------------------
# 📋 FETCH ALL USERS FOR ADMIN
# -----------------------------

@app.route("/api/admin/users", methods=["GET"])
def admin_get_all_users():
    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc:
        return jsonify({"users": []})

    # Flatten all users from batches
    all_users = []
    for batch in users_doc.get("batches", []):
        for user in batch.get("users", []):
            user_copy = user.copy()
            user_copy["batch_name"] = batch.get("batch_name")
            all_users.append(user_copy)

    return jsonify({"users": all_users}), 200


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


# -----------------------------
# 🗑️ ADMIN DELETE SPECIFIC USER
# -----------------------------
@app.route("/api/admin/delete-user", methods=["POST"])
def admin_delete_user():
    data = request.json
    user_email = data.get("email", "").lower()  # email of the user to delete

    if not user_email:
        return jsonify({"error": "User email is required"}), 400

    # Get users document
    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc or not users_doc.get("batches"):
        return jsonify({"error": "No users found"}), 404

    user_deleted = False

    # Iterate over batches to find and delete ALL matching users
    for batch in users_doc["batches"]:
        original_count = len(batch["users"])
        batch["users"] = [u for u in batch["users"] if u.get("email", "").lower() != user_email]
        if len(batch["users"]) != original_count:
            user_deleted = True

    if not user_deleted:
        return jsonify({"error": "User not found"}), 404

    # Save updated users back to MongoDB
    users_collection.replace_one({"_id": "users"}, users_doc, upsert=True)

    return jsonify({"message": f"User {user_email} deleted successfully"}), 200




@app.route("/api/admin/users-by-date", methods=["GET"])
def admin_users_by_date():
    """
    Return a list of all dates and users grouped by created_at date.
    Response:
      {
        "dates": ["All", "2025-11-14", "2025-11-13", ...],
        "usersByDate": {
           "All": [...],
           "2025-11-14": [...],
           "2025-11-13": [...],
           ...
        }
      }
    """
    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc or not users_doc.get("batches"):
        return jsonify({"dates": ["All"], "usersByDate": {"All": []}}), 200

    all_users = []
    for batch in users_doc.get("batches", []):
        for user in batch.get("users", []):
            user_copy = user.copy()
            user_copy["batch_name"] = batch.get("batch_name")
            all_users.append(user_copy)

    # Sort all users by created_at (newest first)
    all_users.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    # Group by created_at date (extract date part only: YYYY-MM-DD)
    users_by_date = {}
    users_by_date["All"] = all_users

    for user in all_users:
        created_at = user.get("created_at")
        if created_at:
            # Extract date part (YYYY-MM-DD format)
            if isinstance(created_at, str):
                date_str = created_at.split("T")[0]  # "2025-11-14T11:39:42..." -> "2025-11-14"
            else:
                # If it's a datetime object
                date_str = created_at.strftime("%Y-%m-%d")
        else:
            date_str = "Unknown Date"
        
        if date_str not in users_by_date:
            users_by_date[date_str] = []
        users_by_date[date_str].append(user)

    # Build dates list in order: All first, then sorted dates (newest first)
    extra_dates = [d for d in users_by_date.keys() if d != "All"]
    # Sort dates in descending order (newest first), Unknown Date at end
    sorted_dates = sorted([d for d in extra_dates if d != "Unknown Date"], reverse=True)
    if "Unknown Date" in extra_dates:
        sorted_dates.append("Unknown Date")
    dates_list = ["All"] + sorted_dates

    return jsonify({"dates": dates_list, "usersByDate": users_by_date}), 200


# Update existing export endpoint to accept optional createdDate query param:
@app.route("/api/admin/export-users", methods=["GET"])
def export_users():
    # optional createdDate query param (YYYY-MM-DD format)
    date_filter = request.args.get("createdDate")  # can be None or "All" or a specific date

    users_doc = users_collection.find_one({"_id": "users"})
    if not users_doc or not users_doc.get("batches"):
        return jsonify({"error": "No users found"}), 404

    # Flatten users data and optionally filter by created_at date
    all_users = []
    for batch in users_doc["batches"]:
        for user in batch["users"]:
            created_at = user.get("created_at")
            
            # Extract date part for comparison
            if created_at:
                if isinstance(created_at, str):
                    user_date_str = created_at.split("T")[0]
                else:
                    user_date_str = created_at.strftime("%Y-%m-%d")
            else:
                user_date_str = "Unknown Date"

            # Apply filter if provided and not "All"
            if date_filter and date_filter != "All" and date_filter != user_date_str:
                continue

            cohort = user.get("cohortinformation", {}) or {}
            program_name = cohort.get("programName") or "Unknown Program"

            flat_user = {
                "fullName": user.get("fullName"),
                "email": user.get("email"),
                "joined_date": user_date_str,
                "programName": program_name,
                "created_at": user.get("created_at"),
            }

            # Cohort information (kept)
            flat_user["programDates"] = cohort.get("programDates")
            flat_user["programVenue"] = cohort.get("programVenue")

            # Demographics
            demo = user.get("demographics", {}) or {}
            for key, value in demo.items():
                flat_user[f"demographics_{key}"] = value

            # Survey
            survey = user.get("survey", {}) or {}
            for key, value in survey.items():
                flat_user[f"survey_{key}"] = value

            # Dreamteam
            dreamteam = user.get("dreamteam", {}) or {}
            for key, value in dreamteam.items():
                flat_user[f"dreamteam_{key}"] = value

            # BigFive
            bigfive = user.get("bigfive", {}) or {}
            for key, value in bigfive.items():
                flat_user[f"bigfive_{key}"] = value

            all_users.append(flat_user)

    # Sort by created_at (newest first)
    all_users.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    # Convert to DataFrame
    df = pd.DataFrame(all_users)

    # Save to Excel in memory
    output = io.BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)

    # Build download_name to indicate date when filtered
    download_name = "all_users.xlsx"
    if date_filter and date_filter != "All":
        download_name = f"users_{date_filter}.xlsx"

    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=download_name
    )
 
 

@app.route("/api/media/profile/<filename>", methods=["GET"])
def get_profile_photo(filename):
    return send_from_directory(PROFILE_FOLDER, filename)

@app.route("/api/media/cover/<filename>", methods=["GET"])
def get_cover_photo(filename):
    return send_from_directory(COVER_FOLDER, filename)



# -----------------------------------------
# 🚀 RUN SERVER
# -----------------------------------------

# --- Production vs. Development Startup ---
if __name__ == '__main__':
    # This block is only for local development testing. 
    # Render's Gunicorn command will ignore this block.
    # We ensure it binds correctly if run locally, though 
    # typically you don't run it this way in production.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)