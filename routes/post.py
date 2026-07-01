from flask import Blueprint, request, jsonify
from datetime import datetime
from bson import ObjectId
from bson.errors import InvalidId
from utils.jwt_auth import token_required

# assume you already have db connection
# from config import db   # adjust if your db import is different
from config.db import db


post_bp = Blueprint("post", __name__)
 
@post_bp.route("/posts", methods=["POST"])
@token_required
def create_post(current_user_email):
    data = request.json

    text = data.get("text", "").strip()
    images = data.get("images", [])

    if len(text) > 300:
        return jsonify({"error": "Text too long"}), 400

    if len(images) > 3:
        return jsonify({"error": "Max 3 images allowed"}), 400

    post = {
        "author_email": current_user_email,
        "text": text,
        "images": images,
        "created_at": datetime.utcnow()
    }

    result = db.posts.insert_one(post)

    return jsonify({
        "message": "Post created",
        "post_id": str(result.inserted_id)
    }), 201


@post_bp.route("/posts/<post_id>/like", methods=["POST"])
@token_required
def toggle_like(current_user_email, post_id):

    like_query = {
        "post_id": ObjectId(post_id),
        "user_email": current_user_email
    }

    existing_like = db.likes.find_one(like_query)

    if existing_like:
        db.likes.delete_one({"_id": existing_like["_id"]})
        return jsonify({"liked": False}), 200

    db.likes.insert_one(like_query)
    return jsonify({"liked": True}), 201


@post_bp.route("/posts/<post_id>", methods=["DELETE"])
@token_required
def delete_post(current_user_email, post_id):
    try:
        post_obj_id = ObjectId(post_id)
    except (InvalidId, TypeError):
        return jsonify({"error": "Invalid post id"}), 400

    post = db.posts.find_one({"_id": post_obj_id})

    if not post:
        return jsonify({"error": "Post not found"}), 404

    if post["author_email"] != current_user_email:
        return jsonify({"error": "Not authorized to delete this post"}), 403

    db.posts.delete_one({"_id": post_obj_id})
    db.likes.delete_many({"post_id": post_obj_id})

    return jsonify({"message": "Post deleted"}), 200


@post_bp.route("/feed", methods=["GET"])
@token_required
def get_feed(current_user_email):

    # 🔹 TEMP: fetch all posts (will filter by matches later)
    posts = list(
        db.posts.find().sort("created_at", -1)
    )

    feed = []

    for post in posts:
        # count likes
        like_count = db.likes.count_documents({
            "post_id": post["_id"]
        })

        # check if current user liked this post
        liked_by_me = db.likes.find_one({
            "post_id": post["_id"],
            "user_email": current_user_email
        }) is not None

        feed.append({
            "post_id": str(post["_id"]),
            "author_email": post["author_email"],
            "text": post["text"],
            "images": post["images"],
            "created_at": post["created_at"],
            "like_count": like_count,
            "liked_by_me": liked_by_me
        })

    return jsonify(feed), 200
