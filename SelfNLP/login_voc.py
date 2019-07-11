from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import os
from flask import Flask, flash, request, redirect, render_template,send_file
from werkzeug.utils import secure_filename
from preprocess_VOC import preprocess_text
import pandas as pd 
import xlrd

ALLOWED_EXTENSIONS = set(['txt', 'csv', 'xlsx'])
app = Flask(__name__)

app.config.from_object("config")
# Change this to your secret key (can be anything, it's for extra protection)
# app.secret_key = 'your_secret_key'

# Enter your database connection details below
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'nlp'
UPLOAD_FOLDER = '/home/ubuntu/data'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Intialize MySQL
mysql = MySQL(app)
@app.route('/home')
def home():
    # Check if user is loggedin
	name_output = os.path.join(UPLOAD_FOLDER+"/"+session['username']+"/output/" )
	list_name = os.listdir(name_output)
	flash("The files you currently have are:")
	i = 1
	for names in list_name:
		flash(names)
		i = i+1
	if 'loggedin' in session:
        # User is loggedin show them the home page
		return render_template('home.html', username=session['username'])
    # User is not loggedin redirect to login page
	return redirect(url_for('login'))
    

@app.route('/', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password))
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    # Show the login form with message (if any)
    return render_template('index.html', msg=msg)


@app.route('/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', [username])
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
            filename = UPLOAD_FOLDER+"/"+username
            print(filename)
            os.makedirs(filename)
            os.makedirs(filename+"/input")
            os.makedirs(filename+"/output")
            
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)

@app.route('/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', [session['id']])
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))
	
@app.route('/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

	
def preprocessing(df,columnname,filename):
    features_file,actions_file,feelings_file,noun_adj_file = preprocess_text(df,columnname)
    features_file.to_excel(filename+"/features_file.xlsx",index = False)
    feelings_file.to_excel(filename+"/feelings_file.xlsx",index = False)
    noun_adj_file.to_excel(filename+"/noun_adj_file.xlsx",index = False)
    actions_file.to_excel(filename+"/actions_file.xlsx",index = False)
	
@app.route('/upload', methods=['POST'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(url_for('home'))
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(url_for('home'))
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			#eliminating the .xlsx and other extensions from filename
			filename_name = str(filename)
			filename_name = filename_name[0:filename_name.find('.')]
			file.save(os.path.join(UPLOAD_FOLDER+"/"+session['username']+"/input", filename))
			flash('File successfully uploaded')
			name_input = os.path.join(UPLOAD_FOLDER+"/"+session['username']+"/input", filename)
			name_output = os.path.join(UPLOAD_FOLDER+"/"+session['username']+"/output/", filename_name )
			try:  
				os.mkdir(name_output)
			except OSError:  
				print ("The directory %s already exists" % path)
			df = pd.read_excel(name_input)
			preprocessing(df,'review',name_output)
			return redirect(url_for('home'))
		else:
			flash('Allowed file types are txt, csv, xlsx')
			return redirect(url_for('home'))

        
if __name__ == "__main__":
	app.run(port = 8989, debug=True)

        
