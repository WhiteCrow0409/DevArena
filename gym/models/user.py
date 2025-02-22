from datetime import datetime
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

client = MongoClient('mongodb://localhost:27017')
db = client['fitframe']
users = db.users
exercise_records = db.exercise_records
rewards = db.rewards

class User:
    def __init__(self, username, email, password=None):
        self.username = username
        self.email = email
        self.password_hash = generate_password_hash(password) if password else None
        self.profile_pic = "default.jpg"
        self.total_points = 0
        self.daily_points = 0
        self.last_points_update = datetime.now().date()

    @staticmethod
    def create_user(username, email, password):
        if users.find_one({"$or": [{"username": username}, {"email": email}]}):
            return False
        
        user = User(username, email, password)
        users.insert_one({
            "username": user.username,
            "email": user.email,
            "password_hash": user.password_hash,
            "profile_pic": user.profile_pic,
            "total_points": user.total_points,
            "daily_points": user.daily_points,
            "last_points_update": user.last_points_update
        })
        return True

    @staticmethod
    def get_user(username):
        user_data = users.find_one({"username": username})
        if not user_data:
            return None
        return user_data

    @staticmethod
    def verify_password(username, password):
        user_data = User.get_user(username)
        if not user_data:
            return False
        return check_password_hash(user_data['password_hash'], password)

    @staticmethod
    def update_points(username, accuracy):
        user_data = User.get_user(username)
        if not user_data:
            return False

        today = datetime.now().date()
        last_update = user_data.get('last_points_update', today)

        # Reset daily points if it's a new day
        if today != last_update:
            users.update_one(
                {"username": username},
                {"$set": {"daily_points": 0, "last_points_update": today}}
            )
            user_data['daily_points'] = 0

        # Add point if accuracy is >= 96% and daily limit not reached
        if accuracy >= 96 and user_data['daily_points'] < 10:
            users.update_one(
                {"username": username},
                {
                    "$inc": {
                        "daily_points": 1,
                        "total_points": 1
                    }
                }
            )
            return True
        return False

class ExerciseRecord:
    @staticmethod
    def add_record(username, exercise_type, reps, accuracy, points_earned):
        exercise_records.insert_one({
            "username": username,
            "exercise_type": exercise_type,
            "reps": reps,
            "accuracy": accuracy,
            "points_earned": points_earned,
            "date": datetime.now()
        })

    @staticmethod
    def get_monthly_records(username):
        month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return list(exercise_records.find({
            "username": username,
            "date": {"$gte": month_start}
        }).sort("date", -1))

class Reward:
    @staticmethod
    def initialize_rewards():
        if rewards.count_documents({}) == 0:
            default_rewards = [
                {"name": "1-Month Free Membership", "points_required": 100, "description": "Get one month free gym membership"},
                {"name": "Personal Training Session", "points_required": 50, "description": "One free personal training session"},
                {"name": "Protein Shake", "points_required": 20, "description": "Free protein shake at the gym"},
                {"name": "Water Bottle", "points_required": 30, "description": "FitFrame branded water bottle"},
                {"name": "Gym Towel", "points_required": 25, "description": "High-quality gym towel"}
            ]
            rewards.insert_many(default_rewards)

    @staticmethod
    def get_all_rewards():
        return list(rewards.find())

    @staticmethod
    def can_redeem(username, reward_points):
        user_data = User.get_user(username)
        return user_data and user_data['total_points'] >= reward_points
