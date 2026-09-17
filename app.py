import os
import re
import time
import pandas as pd
import PyPDF2
from datetime import datetime, timedelta

from flask import Flask, render_template, request, redirect, session
from flask_sqlalchemy import SQLAlchemy
from sentence_transformers import SentenceTransformer, util
from werkzeug.security import generate_password_hash, check_password_hash

# =========================
# APP CONFIG
# =========================
app = Flask(__name__)
app.secret_key = "secret123"

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///hirescope.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Upload folder
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =========================
# DATABASE MODELS
# =========================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    role = db.Column(db.String(50))  # candidate / recruiter


class Resume(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    file_path = db.Column(db.String(200))
    extracted_text = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# =========================
# LOAD MODEL
# =========================
print("Loading AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# =========================
# PDF TEXT EXTRACTION
# =========================
def extract_text_from_pdf(path):
    text = ""
    try:
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += (page.extract_text() or "") + " "
    except Exception as e:
        print("PDF Error:", e)
    return text

# =========================
# ANALYSIS FUNCTIONS
# =========================
def analyze_with_jd(resume_text, job_desc):
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    jd_emb = model.encode(job_desc, convert_to_tensor=True)

    similarity = util.cos_sim(resume_emb, jd_emb)[0][0].item()

    jd_skills = [s.strip().lower() for s in re.split(r',|\n', job_desc) if s.strip()]
    resume_text_lower = resume_text.lower()

    matched = [s for s in jd_skills if s in resume_text_lower]
    skill_ratio = len(matched) / len(jd_skills) if jd_skills else 0

    final_score = (0.6 * similarity) + (0.4 * skill_ratio)

    return int(final_score * 100), matched


def get_detailed_analysis(resume_text, df):
    job_profiles = (df['Title'] + " " + df['Skills'] + " " + df['Responsibilities']).tolist()

    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    job_embs = model.encode(job_profiles, convert_to_tensor=True)
    cosine_scores = util.cos_sim(resume_emb, job_embs)[0].cpu().tolist()

    squashed_resume = "".join(resume_text.lower().split())
    final_results = []

    for i, row in df.iterrows():
        job_skills = str(row['Skills']).split(';')

        matched = [
            s.strip() for s in job_skills
            if "".join(s.lower().split()) in squashed_resume
        ]

        skill_ratio = len(matched) / (len(job_skills) or 1)
        score = (cosine_scores[i] * 0.6) + (skill_ratio * 0.4)

        final_results.append({
            'title': row['Title'],
            'score': int(score * 100),
            'matched_skills': matched,
            'all_skills': job_skills
        })

    final_results.sort(key=lambda x: x['score'], reverse=True)
    top = final_results[0]

    missing = [
        s.strip() for s in top['all_skills']
        if s.strip() and s.strip() not in top['matched_skills']
    ]

    # Extract info
    lines = [l.strip() for l in resume_text.split('\n') if l.strip()]
    name = lines[0] if lines else "Candidate"
    name = re.split(r'Email|Phone|LinkedIn', name, flags=re.IGNORECASE)[0].strip()

    phone_match = re.search(r'(\+91[\s-]?\d{10}|\d{10})', resume_text)
    phone = phone_match.group(0) if phone_match else "Not found"

    li_match = re.search(r'linkedin\.com/in/[\w-]+', resume_text)
    linkedin = li_match.group(0) if li_match else "#"

    return {
        'candidate_name': name,
        'phone': phone,
        'linkedin': linkedin,
        'top_match': top,
        'matched_skills': top['matched_skills'],
        'missing_skills': missing,
        'others': final_results[1:6]
    }

# =========================
# AUTH ROUTES
# =========================
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        if User.query.filter_by(email=request.form['email']).first():
            return "Email already exists"

        user = User(
            name=request.form['name'],
            email=request.form['email'],
            password=generate_password_hash(request.form['password']),
            role=request.form['role']
        )
        db.session.add(user)
        db.session.commit()
        return redirect('/login')

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()

        if user and check_password_hash(user.password, request.form['password']):
            session['user_id'] = user.id
            session['role'] = user.role

            return redirect('/recruiter') if user.role == "recruiter" else redirect('/')

        return "Invalid credentials"

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

# =========================
# MAIN (CANDIDATE)
# =========================
@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user_id' not in session:
        return redirect('/login')

    if session.get('role') != 'candidate':
        return redirect('/recruiter')

    analysis = None

    if request.method == 'POST':
        file = request.files.get('resume')

        if file:
            filename = str(int(time.time()*1000)) + "_" + file.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            text = extract_text_from_pdf(path)
            df = pd.read_csv('job_dataset.csv').fillna('N/A')
            analysis = get_detailed_analysis(text, df)

            db.session.add(Resume(
                user_id=session['user_id'],
                file_path=path,
                extracted_text=text
            ))
            db.session.commit()

    return render_template('index.html', analysis=analysis)

# =========================
# RECRUITER
# =========================
@app.route("/recruiter", methods=["GET", "POST"])
def recruiter():
    if 'user_id' not in session:
        return redirect('/login')

    if session.get('role') != 'recruiter':
        return "Access Denied"

    results = []

    if request.method == "POST":
        files = request.files.getlist("resumes")
        job_desc = request.form.get("job_desc")

        for file in files:
            if file.filename == "":
                continue

            filename = str(int(time.time()*1000)) + "_" + file.filename
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            text = extract_text_from_pdf(path)

            # Save in DB
            db.session.add(Resume(
                user_id=session['user_id'],
                file_path=path,
                extracted_text=text
            ))

            if job_desc:
                score, _ = analyze_with_jd(text, job_desc)
                role_name = "Custom JD"
            else:
                df = pd.read_csv('job_dataset.csv').fillna('N/A')
                analysis = get_detailed_analysis(text, df)
                score = analysis['top_match']['score']
                role_name = analysis['top_match']['title']

            results.append({
                "name": text.split('\n')[0][:30],
                "score": score,
                "role": role_name
            })

        db.session.commit()
        results.sort(key=lambda x: x['score'], reverse=True)

    return render_template("recruiter.html", results=results)

# =========================
# HISTORY (SHARED)
# =========================
@app.route("/history")
def history():
    if 'user_id' not in session:
        return redirect('/login')

    resumes = Resume.query.filter_by(user_id=session['user_id'])\
                          .order_by(Resume.created_at.desc())\
                          .all()

    return render_template("history.html", resumes=resumes, timedelta=timedelta)

# =========================
# VIEW RESUME
# =========================
@app.route("/view/<int:resume_id>")
def view_resume(resume_id):
    if 'user_id' not in session:
        return redirect('/login')

    resume = Resume.query.get_or_404(resume_id)

    if resume.user_id != session['user_id']:
        return "Access Denied"

    df = pd.read_csv('job_dataset.csv').fillna('N/A')
    analysis = get_detailed_analysis(resume.extracted_text, df)

    return render_template("index.html", analysis=analysis)

# =========================
# INIT DB
# =========================
with app.app_context():
    db.create_all()
    print(" Database ready!")

# =========================
#  RUN
# =========================
if __name__ == '__main__':
    app.run(debug=True)