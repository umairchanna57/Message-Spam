from flask import Flask, render_template, url_for, redirect, request
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

app = Flask(__name__)
login_manager = LoginManager(app)
bcrypt = Bcrypt(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SECRET_KEY'] = 'this is a secret key '
db = SQLAlchemy(app)


class  User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    # is_active = db.Column(db.Boolean(), default=True)

    def __repr__(self):
        return f'<User {self.username}>'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/welcome')
def welcome():
    return render_template('welcome.html')



@app.route('/login', methods=[ 'GET', 'POST'])
def login():
    if request.method =='POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and bcrypt.check_password_hash(user.password , password):
            login_user(user)
            return redirect(url_for('welcome'))
    return render_template('login.html')



@app.route('/register'  , methods=['GET', 'POST'])
def register():
    if request.method =='POST':
        username = request.form['username']
        password= request.form['password']
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')    #.decode('utf-8') part converts the byte string into a UTF-8 encoded string

        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('registeration.html')






@app.route('/logout_page')
def logout_page():
    logout_user()
    return render_template('logout.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('logout_page'))


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
