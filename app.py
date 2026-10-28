from flask import Flask, render_template, request, redirect, session, flash, send_file, url_for
import sqlite3
import numpy as np
import joblib
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import os
from dotenv import load_dotenv
from google import genai
import pandas as pd

# ---------------------------
# LOAD ENV VARIABLES
# ---------------------------
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_KEY)

app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = "static/prescriptions"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ---------------------------
# LOAD FOOD CSV
# ---------------------------
food_df = None
try:
    food_df = pd.read_csv("data/food.csv")
except:
    print("food.csv missing in /data folder")

# ---------------------------
# DATABASE INITIAL SETUP
# ---------------------------
def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT DEFAULT 'user'
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            result TEXT,
            pregnancies REAL,
            glucose REAL,
            bp REAL,
            skin REAL,
            insulin REAL,
            bmi REAL,
            dpf REAL,
            age REAL,
            date TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            medicine_name TEXT,
            dosage TEXT,
            reminder_time TEXT,
            reminder_date TEXT,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prescriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            visit_date TEXT,
            doctor_note TEXT,
            file_path TEXT,
            uploaded_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)


    # Create admin
    cursor.execute("SELECT * FROM users WHERE role='admin'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                       ("admin", generate_password_hash("admin123"), "admin"))

    conn.commit()
    conn.close()

init_db()

# ---------------------------
# LOAD MODEL
# ---------------------------
model = joblib.load("model/diabetes_model.pkl")
scaler = joblib.load("model/scaler.pkl")

heart_model = joblib.load("model/heart_model.pkl")
heart_scaler = joblib.load("model/heart_scaler.pkl")

# ============================================================
# STRONG FEATURE 1: FUTURE DIABETES RISK FORECASTING
# ============================================================
def future_risk_forecast(user_id):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT glucose, bmi 
        FROM history
        WHERE user_id=?
        ORDER BY id ASC
    """, (user_id,))

    records = cursor.fetchall()
    conn.close()

    if len(records) < 3:
        return "Not enough historical data for future risk analysis"

    glucose_trend = records[-1][0] - records[0][0]
    bmi_trend = records[-1][1] - records[0][1]

    if glucose_trend > 20 or bmi_trend > 3:
        return "HIGH RISK (Worsening condition)"
    elif glucose_trend > 10 or bmi_trend > 1.5:
        return "MODERATE RISK"
    else:
        return "LOW RISK (Stable condition)"


# ============================================================
# STRONG FEATURE 2: EXPLAINABLE AI (XAI)
# ============================================================
def explain_prediction(input_data):
    feature_names = [
        "Pregnancies",
        "Glucose",
        "Blood Pressure",
        "Skin Thickness",
        "Insulin",
        "BMI",
        "Diabetes Pedigree Function",
        "Age"
    ]

    values = np.abs(input_data[0])
    total = np.sum(values)

    contributions = (values / total) * 100
    explanation = dict(zip(feature_names, np.round(contributions, 2)))

    return explanation


# ---------------------------
# ROUTES
# ---------------------------

@app.route('/heart', methods=['GET','POST'])
def heart():

    if "user_id" not in session:
        return redirect('/login')

    if request.method == 'POST':

        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])

        data = np.array([[age,sex,cp,trestbps,chol,fbs,
                          restecg,thalach,exang,oldpeak,
                          slope,ca,thal]])

        scaled = heart_scaler.transform(data)

        prediction = heart_model.predict(scaled)[0]

        result = "Heart Disease Detected" if prediction==1 else "No Heart Disease"

        return render_template("heart_result.html",result=result)

    return render_template("heart.html")

@app.route('/choose')
def choose():

    if "user_id" not in session:
        return redirect('/login')

    return render_template("choose.html")

@app.route('/')
def home():
    return redirect('/login')

# ---------------------------
# USER ROUTES
# ---------------------------
@app.route('/index', methods=['GET', 'POST'])
def index():
    if "user_id" not in session or session['role'] != 'user':
        return redirect('/login')

    if request.method == 'POST':
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        bp = float(request.form['bp'])
        skin = float(request.form['skin'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = float(request.form['age'])

        input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"

        # ---------------------------
# NEW STRONG FEATURES
# ---------------------------
        future_risk = future_risk_forecast(session['user_id'])
        explanation = explain_prediction(input_data)
        risk_probability = round(model.predict_proba(scaled_data)[0][1] * 100, 2)


        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO history 
            (user_id, result, pregnancies, glucose, bp, skin, insulin, bmi, dpf, age, date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session['user_id'], result, pregnancies, glucose, bp, skin, insulin, bmi, dpf, age,
            datetime.now().strftime("%Y-%m-%d %H:%M")
        ))
        conn.commit()
        conn.close()

        return render_template(
            "result.html",
            result=result,
            pregnancies=pregnancies,
            glucose=glucose,
            bp=bp,
            skin=skin,
            insulin=insulin,
            bmi=bmi,
            dpf=dpf,
            age=age,
            future_risk=future_risk,
            explanation=explanation,
            risk_probability=risk_probability

        )

    return render_template("index.html")


@app.route('/dashboard')
def dashboard():
    if "user_id" not in session or session['role'] != 'user':
        return redirect('/login')

    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history WHERE user_id=? ORDER BY date DESC", (session['user_id'],))
    records = cursor.fetchall()
    conn.close()

    return render_template("dashboard.html", history=records)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed = generate_password_hash(password)

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?,?)", (username, hashed))
            conn.commit()
            flash("Registration successful!", "success")
            return redirect('/login')
        except:
            flash("Username already exists", "danger")
        conn.close()

    return render_template("register.html")

@app.route("/upload_prescription", methods=["GET", "POST"])
def upload_prescription():
    if "user_id" not in session:
        return redirect("/login")

    if request.method == "POST":
        visit_date = request.form["visit_date"]
        doctor_note = request.form["doctor_note"]
        file = request.files["prescription"]

        if file:
            filename = f"{session['user_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            conn = sqlite3.connect("users.db")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO prescriptions
                (user_id, visit_date, doctor_note, file_path, uploaded_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session["user_id"],
                visit_date,
                doctor_note,
                file_path,
                datetime.now().strftime("%Y-%m-%d %H:%M")
            ))
            conn.commit()
            conn.close()

            flash("Prescription uploaded successfully", "success")
            return redirect("/view_prescriptions")

    return render_template("upload_prescription.html")

@app.route("/view_prescriptions")
def view_prescriptions():
    if "user_id" not in session:
        return redirect("/login")

    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM prescriptions
        WHERE user_id=?
        ORDER BY uploaded_at DESC
    """, (session["user_id"],))

    data = cursor.fetchall()
    conn.close()

    return render_template("view_prescriptions.html", prescriptions=data)



@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            session['role'] = user[3]
            return redirect('/admin_dashboard') if user[3]=="admin" else redirect('/choose')

        flash("Invalid login", "danger")

    return render_template("login.html")

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# ---------------------------
# REMINDER SYSTEM
# ---------------------------
@app.route('/reminder', methods=['GET','POST'])
def reminder():
    if "user_id" not in session:
        return redirect('/login')

    if request.method == "POST":
        med = request.form['med_name']
        dose = request.form['dosage']
        time = request.form['time']
        date = request.form['date']

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO reminders (user_id, medicine_name, dosage, reminder_time, reminder_date)
            VALUES (?,?,?,?,?)
        """, (session["user_id"], med, dose, time, date))
        conn.commit()
        conn.close()

        return redirect("/show_reminder")

    return render_template("reminder.html")


@app.route('/show_reminder')
def show_reminder():
    if "user_id" not in session:
        return redirect('/login')

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT medicine_name, dosage, reminder_time, reminder_date, status, id
        FROM reminders
        WHERE user_id=?
    """, (session["user_id"],))
    reminders = cursor.fetchall()
    conn.close()

    return render_template("show_reminder.html", reminders=reminders)

@app.route('/delete_reminder/<int:rid>')
def delete_reminder(rid):
    if "user_id" not in session:
        return redirect('/login')

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM reminders WHERE id=? AND user_id=?",
        (rid, session["user_id"])
    )
    conn.commit()
    conn.close()

    flash("Reminder deleted successfully", "success")
    return redirect('/show_reminder')



# ============================================================
# DIET RECOMMENDATION SYSTEM
# ============================================================
@app.route("/food", methods=["GET"])
def food():
    if "user_id" not in session:
        return redirect("/login")

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT result FROM history WHERE user_id=? ORDER BY id DESC LIMIT 1",
                   (session["user_id"],))
    row = cursor.fetchone()
    conn.close()

    latest_result = row[0] if row else "Non-Diabetic"

    diabetic_recommend = [
        "Whole grains", "Leafy vegetables", "High-fiber fruits", "Lean protein", "Nuts & seeds"
    ]
    diabetic_avoid = [
        "Sugary drinks", "White bread", "Sweets", "Fried foods", "High-carb rice"
    ]

    non_diabetic_recommend = [
        "Balanced meals", "Whole fruits", "Lean meat", "Whole grains", "Vegetables"
    ]
    non_diabetic_avoid = [
        "Excess sugar", "Too much junk food"
    ]

    if latest_result == "Diabetic":
        return render_template("food.html",
                               plan_type="Your Diet Recommendation (Diabetic)",
                               foods=diabetic_recommend,
                               avoid=diabetic_avoid)
    else:
        return render_template("food.html",
                               plan_type="Your Diet Recommendation (Normal)",
                               foods=non_diabetic_recommend,
                               avoid=non_diabetic_avoid)

@app.route('/report/<int:history_id>')
def report(history_id):
    if "user_id" not in session:
        return redirect('/login')

    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history WHERE id=?", (history_id,))
    record = cursor.fetchone()
    conn.close()

    if not record:
        flash("Report not found!", "danger")
        return redirect('/dashboard')

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # ================= HEADER =================
    c.setFillColor(colors.darkblue)
    c.setFont("Helvetica-Bold", 26)
    c.drawCentredString(width / 2, height - 60, "DIABETES TEST REPORT")

    c.setStrokeColor(colors.grey)
    c.setLineWidth(1)
    c.line(50, height - 80, width - 50, height - 80)

    # ================= PATIENT INFO =================
    y = height - 130
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.black)
    c.drawString(50, y, "Patient Test Details")
    y -= 15

    c.setLineWidth(0.5)
    c.line(50, y, width - 50, y)
    y -= 30

    # ================= DATA TABLE =================
    label_x = 60
    value_x = 260

    c.setFont("Helvetica-Bold", 12)
    c.drawString(label_x, y, "Parameter")
    c.drawString(value_x, y, "Value")
    y -= 10
    c.line(50, y, width - 50, y)
    y -= 25

    c.setFont("Helvetica", 12)

    fields = [
        ("Result", record["result"]),
        ("Pregnancies", record["pregnancies"]),
        ("Glucose Level", record["glucose"]),
        ("Blood Pressure", record["bp"]),
        ("Skin Thickness", record["skin"]),
        ("Insulin", record["insulin"]),
        ("BMI", record["bmi"]),
        ("Diabetes Pedigree Function", record["dpf"]),
        ("Age", record["age"]),
        ("Test Date", record["date"]),
    ]

    for label, value in fields:
        if label == "Result":
            c.setFont("Helvetica-Bold", 13)
            if value == "Diabetic":
                c.setFillColor(colors.red)
            else:
                c.setFillColor(colors.green)
        else:
            c.setFont("Helvetica", 12)
            c.setFillColor(colors.black)

        c.drawString(label_x, y, label)
        c.drawString(value_x, y, str(value))
        y -= 28

    # ================= FOOTER =================
    c.setStrokeColor(colors.grey)
    c.line(50, 80, width - 50, 80)

    c.setFont("Helvetica-Oblique", 10)
    c.setFillColor(colors.grey)
    c.drawCentredString(
        width / 2,
        60,
        "Generated by Diabetes Prediction System"
    )

    c.showPage()
    c.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Diabetes_Test_Report.pdf",
        mimetype="application/pdf"
    )

# ============================================================
# ADMIN PANEL
# ============================================================
@app.route('/admin_dashboard')
def admin_dashboard():
    if "user_id" not in session or session['role'] != 'admin':
        return redirect('/login')

    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT id, username, role FROM users")
    users = cursor.fetchall()
    cursor.execute("""
        SELECT h.id, u.username, h.result, h.date 
        FROM history h 
        JOIN users u ON h.user_id=u.id 
        ORDER BY h.date DESC
    """)
    histories = cursor.fetchall()
    conn.close()

    return render_template("admin.html", users=users, histories=histories)

@app.route('/admin_delete_user/<int:user_id>')
def admin_delete_user(user_id):
    if "user_id" not in session or session['role'] != 'admin':
        return redirect('/login')

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id=?", (user_id,))
    cursor.execute("DELETE FROM history WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()

    flash("User deleted successfully!", "success")
    return redirect('/admin_dashboard')

# ============================================================
# CHATBOT
# ============================================================
@app.route('/chatbot', methods=['GET','POST'])
def chatbot():
    if "user_id" not in session:
        return redirect('/login')

    response_text = ""

    if request.method == "POST":
        msg = request.form['message']
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=msg
            )
            response_text = response.text
        except Exception as e:
            response_text = str(e)

    return render_template("chatbot.html", response=response_text)

# ---------------------------
# RUN APP
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
