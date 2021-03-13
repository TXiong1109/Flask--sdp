#!/usr/bin/env python
from flask import Flask, render_template, session, redirect, request, flash, url_for
import os
import json
from functools import wraps

app = Flask(__name__)
app.config['SECRET_KEY']='aldjfkadq37oaisdfo'
basedir = os.path.dirname(os.path.abspath(__file__))

json_names = ['output-floor1','output-floor2']

def login_check(func):
    @wraps(func)
    def inner(*args, **kwargs):
        if not session.get('username'):
            flash('Please login firstly!')
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    return inner


@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == "POST":
        username = request.form.get('username')
        password = request.form.get('password')
        accounts_file = os.path.join(basedir, 'data/accounts.json')
        with open(accounts_file, 'r', encoding='utf-8') as f:
            accounts = json.load(f)
            if username in accounts['accounts'] and password == accounts['accounts'][username]:
                session['username'] = username
                return redirect(url_for("index"))
            else:
                flash('Username and password not matched!')
                return redirect(url_for("login"))
    return render_template('login.html')


@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == "POST":
        username = request.form.get('username')
        password = request.form.get('password')
        password2 = request.form.get('password2')
        accounts = None
        if password != password2:
            flash('Password not matched, please retry!')
            return render_template('register.html')
        accounts_file = os.path.join(basedir, 'data/accounts.json')
        with open(accounts_file, 'r', encoding='utf-8') as f:
            accounts = json.load(f)
            if username in accounts['accounts']:
                flash('User already exists, pls check!')
                return render_template('register.html')
            accounts['accounts'][username] = password
        if accounts:
            with open(accounts_file, 'w', encoding='utf-8') as f:
                json.dump(accounts, f)
                session['username'] = username
                flash('Register successfully!')
                return render_template('index.html')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route('/')
@login_check
def index():
    jname=""
    return render_template('index.html', json_names = json_names, jname=jname)


@app.route('/json', methods=['GET'])
@login_check
def read_json():
    jname = request.args.get('jname')
    filename = jname + '.json'
    robots = []
    robot  = {}
    print('filepath: --- {}'.format(os.path.join(basedir, 'data' , filename)))
    with open(os.path.join(basedir, 'data' , filename)) as f:
        jsonstr = json.load(f)
        robots = jsonstr.get('Robots')
        # print(robots)
        if robots:
            robot = robots[0]
        return render_template('index.html', json_names = json_names, jname=jname, robots = robots, robot=robot, robotid=0)

@app.route('/<jname>/robot/<int:id>/move', methods=['GET'])
@login_check
def moverobot(jname, id):
    filename = jname + '.json'
    jsonfile = os.path.join(basedir, 'data', filename)
    newjson = None
    with open(jsonfile, 'r', encoding='utf-8') as f:
        jsondata = json.load(f)
        jsondata['Robots'][id]["State"]["State"] = "Movement"
        jsondata['Robots'][id]["State"]["Destination"] = "Reception"
        newjson = jsondata

    with open(jsonfile, 'w', encoding='utf-8') as f:
        json.dump(newjson, f)
        flash('Update robot successfully!')
        return redirect(url_for("robot", jname=jname, id=id))


@app.route('/<jname>/robot/<int:id>', methods=['GET'])
@login_check
def robot(jname, id):
    filename = jname + '.json'
    robots = []
    robot  = {}
    print('filepath: --- {}'.format(os.path.join(basedir, 'data' , filename)))
    with open(os.path.join(basedir, 'data' , filename)) as f:
        jsonstr = json.load(f)
        robots = jsonstr.get('Robots')
        if robots:
            robot = robots[id]
            state = robot["State"]["State"]
            destination = robot["State"]["Destination"]
            location = get_location(state, destination)
            print('locaiton: {}'.format(location))
        return render_template('index.html', json_names = json_names, jname=jname, robots = robots, robot=robot, location=location, robotid=id)

def get_location(state, destination):
    print('state: {}, destination: {}'.format(state, destination))
    location = 'Reception'
    if state == "WaitingAtReception":
        if destination == "N/A":
            location = "Reception"
    elif state == "ReadyForMovement":
        if destination == "Booth":
            location = "Ready to move to Booth"
        if destination == "Reception":
            location = "Ready to move to Reception"
        if destination == "Exit":
            location = "Ready to move to Exit"
        if destination == "Drop off":
            location = "Ready to move to Dropoff"
    elif state == "Charging":
        if destination == "N/A":
            location = "Charging Center"
    elif state == "Movement":
        if destination == "Booth":
            location = "Way to Booth"
        if destination == "Reception":
            location = "Way to Reception"
        if destination == "Exit":
            location = "Way to Exit"
        if destination == "DropOff":
            location = "Way to Dropoff"
    elif state == "AtExit":
        if destination == "N/A":
            location = "Exit"
    elif state == "AtDropOff":
        if destination == "N/A":
            location = "Dropoff"
    elif state == "WaitingForAssistance":
        if destination == "N/A":
            location = "Booth"
    return location





if __name__ == '__main__':
    app.run(debug=True,port=8000)
