from app import app, db, User

with app.app_context():
    db.create_all()