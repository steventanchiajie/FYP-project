import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
import csv
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = 'admin'

def read_data():
    data = []
    with open('MalaysiaNews.csv', 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    return data

def write_data(data):
    fieldnames = ['title', 'content', 'boolean', 'year', 'url']
    try:
        with open('MalaysiaNews.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
    except PermissionError as e:
        print(f"Permission Error: Unable to write to the file. Details: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print("Please check file permissions and location.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    data = read_data()
    return render_template('adminindex.html', news=data)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'admin123':
            session['logged_in'] = True
            flash('Logged in successfully.', 'success')
            return redirect(url_for('admin'))
        else:
            flash('Invalid credentials', 'error')
            return render_template('adminlogin.html', error='Invalid credentials')
    return render_template('adminlogin.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        new_entry = {
            'title': request.form['title'],
            'content': request.form['content'],
            'boolean': request.form['boolean'],
            'year': request.form['year'],
            'url': request.form['url']
        }
        data = read_data()
        data.append(new_entry)
        write_data(data)
        flash('New entry added successfully.', 'success')
        return redirect(url_for('admin'))
    return render_template('admin.html')

if __name__ == '__main__':
    app.run(debug=True)