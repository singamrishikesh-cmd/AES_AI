from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from flask_mysqldb import MySQL
import sys
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
import json
from collections import defaultdict
import os
import re
from werkzeug.utils import secure_filename
import PyPDF2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image
import cv2
import numpy as np
from pdf2image import convert_from_path
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
import tempfile
from io import BytesIO
from xml.sax.saxutils import escape as xml_escape
from datetime import datetime

# PDF generation import
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    HAS_REPORTLAB = True
except ImportError:
    print("Warning: reportlab not installed. PDF export will be disabled. Install with: pip install reportlab")
    HAS_REPORTLAB = False

warnings.filterwarnings("ignore")

# Download NLTK resources with error handling
try:
    nltk.download("stopwords", quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK stopwords: {e}")

try:
    nltk.download("punkt", quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK punkt: {e}")

try:
    nltk.download("punkt_tab", quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK punkt_tab: {e}")

try:
    nltk.download("wordnet", quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK wordnet: {e}")

try:
    nltk.download("words", quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK words corpus: {e}")

try:
    nltk.download("vader_lexicon", quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK vader_lexicon: {e}")

# Load SentenceTransformer model once for performance
sentence_model = None
USE_SENTENCE_TRANSFORMER = False

# Try to load SentenceTransformer (moved to avoid module-level import issues)
def load_sentence_transformer():
    global sentence_model, USE_SENTENCE_TRANSFORMER
    if sentence_model is None and not USE_SENTENCE_TRANSFORMER:
        try:
            from sentence_transformers import SentenceTransformer
            sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            USE_SENTENCE_TRANSFORMER = True
            print("SentenceTransformer model loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load SentenceTransformer: {e}")
            sentence_model = None
            USE_SENTENCE_TRANSFORMER = False
    return USE_SENTENCE_TRANSFORMER and sentence_model is not None

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Set the template folder
app.template_folder = 'templates'

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''  # Enter your MySQL password here
app.config['MYSQL_DB'] = 'teacher_part'

mysql = MySQL(app)

# File upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'txt'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set English stopwords
EN_STOPWORDS = set(stopwords.words("english"))

# Preprocess text
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
import language_tool_python

def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)  # Tokenization
    # Try lemmatization, fallback to original tokens if NLTK fails
    try:
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    except Exception as e:
        print(f"Lemmatization failed: {e}, using original tokens")
        lemmatized_tokens = [token for token in tokens if token.isalnum()]
    return lemmatized_tokens

# Exact Match function
def exact_match(expected_answer, student_answer):
    return int(expected_answer == student_answer)

# Partial Match function
def partial_match(expected_answer, student_answer):
    expected_tokens = preprocess_text(expected_answer)
    student_tokens = preprocess_text(student_answer)
    common_tokens = set(expected_tokens) & set(student_tokens)
    match_percentage = len(common_tokens) / max(len(expected_tokens), len(student_tokens))
    return match_percentage

# Cosine Similarity function
def cosine_similarity_score(expected_answer, student_answer):
    try:
        vectorizer = TfidfVectorizer(tokenizer=preprocess_text)
        tfidf_matrix = vectorizer.fit_transform([expected_answer, student_answer])

        # if vectorizer found no tokens, return 0
        if tfidf_matrix.shape[1] == 0:
            return 0.0

        cosine_sim_val = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        if np.isnan(cosine_sim_val) or not np.isfinite(cosine_sim_val):
            return 0.0
        return float(cosine_sim_val)
    except Exception as e:
        print(f"Cosine similarity error: {e}")
        return 0.0

# Sentiment Analysis function
def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return float((sentiment_score + 1) / 2)  # Normalize to range [0, 1]

# Function to calculate enhanced sentence match score using Semantic Similarity
def enhanced_sentence_match(expected_answer, student_answer):
    load_sentence_transformer()  # Ensure model is loaded
    if USE_SENTENCE_TRANSFORMER and sentence_model:
        try:
            # Truncate text to first 512 words for faster processing
            expected_truncated = ' '.join(expected_answer.split()[:512])
            student_truncated = ' '.join(student_answer.split()[:512])
            
            embeddings_expected = sentence_model.encode([expected_truncated])
            embeddings_student = sentence_model.encode([student_truncated])
            similarity = cosine_similarity([embeddings_expected.flatten()], [embeddings_student.flatten()])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Sentence transformer error: {e}, using fallback")
            return float(cosine_similarity_score(expected_answer, student_answer))  # Fallback to TF-IDF similarity
    else:
        return float(cosine_similarity_score(expected_answer, student_answer))  # Fallback to TF-IDF similarity

# Function to calculate multinomial naive Bayes score
def multinomial_naive_bayes_score(expected_answer, student_answer):
    # Convert answers to a list
    answers = [expected_answer, student_answer]

    # Initialize CountVectorizer
    vectorizer = CountVectorizer(tokenizer=preprocess_text)

    # Fit and transform the answers
    X = vectorizer.fit_transform(answers)

    # Labels
    y = [0, 1]  # 0 for expected_answer, 1 for student_answer

    # Train Multinomial Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X, y)

    # Predict probabilities
    probs = clf.predict_proba(X)

    # Return the probability of the student's answer being correct
    return float(probs[1][1])  # Probability of the student's answer being class 1 (student_answer)

# Function to calculate weighted average score
def weighted_average_score(scores, weights):
    weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
    total_weight = sum(weights)
    return weighted_sum / total_weight

def semantic_similarity_score(expected_answer, student_answer):
    load_sentence_transformer()  # Ensure model is loaded
    if USE_SENTENCE_TRANSFORMER and sentence_model:
        try:
            # Truncate text to first 2048 words for faster processing (SentenceTransformer is slow on long texts)
            expected_truncated = ' '.join(expected_answer.split()[:2048])
            student_truncated = ' '.join(student_answer.split()[:2048])
            
            embeddings_expected = sentence_model.encode([expected_truncated])
            embeddings_student = sentence_model.encode([student_truncated])
            similarity = cosine_similarity([embeddings_expected.flatten()], [embeddings_student.flatten()])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Sentence transformer error: {e}, using fallback")
            return float(cosine_similarity_score(expected_answer, student_answer))  # Fallback to TF-IDF similarity
    else:
        return float(cosine_similarity_score(expected_answer, student_answer))  # Fallback to TF-IDF similarity



# Optimized evaluation cache to improve performance
evaluation_cache = {}

def get_cached_evaluation(expected, response, total_marks):
    """Cache evaluation results to improve performance"""
    cache_key = hash((expected, response, total_marks))
    if cache_key in evaluation_cache:
        return evaluation_cache[cache_key]

    result = evaluate_answers(expected, response, total_marks)
    evaluation_cache[cache_key] = result
    return result

def coherence_score(expected_answer, student_answer):
    """Calculate coherence based on length similarity"""
    len_expected = len(word_tokenize(expected_answer))
    len_student = len(word_tokenize(student_answer))
    if len_expected == 0 or len_student == 0:
        return 0.0
    coherence = min(len_expected, len_student) / max(len_expected, len_student)
    return float(coherence)

def relevance_score(expected_answer, student_answer):
    """Calculate relevance based on word overlap"""
    expected_tokens = set(word_tokenize(expected_answer.lower()))
    student_tokens = set(word_tokenize(student_answer.lower()))
    if not expected_tokens:
        return 0.0
    common_tokens = expected_tokens.intersection(student_tokens)
    relevance = len(common_tokens) / len(expected_tokens)
    return float(relevance)

def extract_keywords_improved(expected_answer, student_answer, max_keywords=8):
    """
    Improved keyword extraction using both answers and better TF-IDF scoring
    """
    try:
        # Combine both answers for better keyword context
        combined_text = [expected_answer, student_answer]

        # Use better preprocessing
        vectorizer = TfidfVectorizer(
            tokenizer=lambda x: preprocess_text(x),
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            max_features=100
        )

        tfidf_matrix = vectorizer.fit_transform(combined_text)
        feature_names = vectorizer.get_feature_names_out()

        # Get TF-IDF scores for expected answer
        expected_tfidf = tfidf_matrix[0].toarray().flatten()

        # Sort by TF-IDF score and get top keywords
        top_indices = expected_tfidf.argsort()[-max_keywords:][::-1]
        keywords = []

        for idx in top_indices:
            if expected_tfidf[idx] > 0.1:  # Minimum threshold
                keyword = feature_names[idx]
                # Clean keyword (remove very short words)
                if len(keyword.split()) >= 1 and len(keyword) > 2:
                    keywords.append(keyword.lower())

        # Remove duplicates and limit to max_keywords
        keywords = list(dict.fromkeys(keywords))[:max_keywords]

        return keywords

    except Exception as e:
        print(f"Keyword extraction error: {e}")
        # Fallback: simple word extraction
        expected_tokens = preprocess_text(expected_answer)
        return list(set(expected_tokens))[:max_keywords]

def match_keywords_flexible(keywords, student_text):
    """
    Flexible keyword matching with lemmatization and partial matching
    """
    student_tokens = preprocess_text(student_text)
    student_text_lower = student_text.lower()

    matched = []
    missed = []

    for keyword in keywords:
        keyword_lower = keyword.lower()

        # Exact match in tokens
        if keyword_lower in student_tokens:
            matched.append(keyword)
            continue

        # Partial match in original text (for phrases)
        if keyword_lower in student_text_lower:
            matched.append(keyword)
            continue

        # Check if keyword is contained in any student token
        if any(keyword_lower in token for token in student_tokens):
            matched.append(keyword)
            continue

        # Check for lemmatized matches
        try:
            lemmatizer = WordNetLemmatizer()
            keyword_lemmatized = lemmatizer.lemmatize(keyword_lower)
            if any(lemmatizer.lemmatize(token) == keyword_lemmatized for token in student_tokens):
                matched.append(keyword)
                continue
        except:
            pass

        missed.append(keyword)

    return matched, missed

def calculate_syntactic_similarity(expected, student):
    """
    Calculate syntactic similarity based on sentence structure and grammar
    """
    try:
        # Simple syntactic analysis based on POS tags
        expected_tokens = word_tokenize(expected)
        student_tokens = word_tokenize(student)

        # Get POS tags
        expected_pos = nltk.pos_tag(expected_tokens)
        student_pos = nltk.pos_tag(student_tokens)

        # Compare POS tag sequences (simplified syntactic structure)
        expected_pos_sequence = [tag for _, tag in expected_pos]
        student_pos_sequence = [tag for _, tag in student_pos]

        # Calculate similarity in POS tag sequences
        min_len = min(len(expected_pos_sequence), len(student_pos_sequence))
        if min_len == 0:
            return 0.0

        matches = 0
        for i in range(min_len):
            if expected_pos_sequence[i] == student_pos_sequence[i]:
                matches += 1

        syntactic_sim = matches / max(len(expected_pos_sequence), len(student_pos_sequence))
        return float(syntactic_sim)

    except Exception as e:
        print(f"Syntactic similarity error: {e}")
        return 0.0

def evaluate_answers(expected_answer, student_answer, total_marks):
    """
    Comprehensive evaluation function with improved accuracy and performance
    """

    # Normalize text - preserve meaning better
    def normalize(text):
        if not text:
            return ""
        text = text.lower()
        # Keep more characters - only remove very special ones
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in words if len(word) > 1]
        return ' '.join(stemmed_words)

    expected = normalize(expected_answer)
    student = normalize(student_answer)

    # Handle edge cases
    if not expected and not student:
        return {'score': total_marks, 'details': {'error': 'both answers empty after normalization'}}

    if not expected or not student:
        # If normalization removed all text but raw text contains letters, try fallback using raw string.
        if expected_answer.strip() and student_answer.strip():
            expected = expected_answer.lower().strip()
            student = student_answer.lower().strip()
        else:
            return {'score': 0, 'details': {'error': 'one answer empty after normalization'}}

    # Fast path for exact normalized answers: award full marks immediately
    if expected == student:
        return {
            'score': float(total_marks),
            'details': {
                'cosine_similarity': 1.0,
                'semantic_similarity': 1.0,
                'syntactic_similarity': 1.0,
                'enhanced_sentence_match': 1.0,
                'keyword_coverage': 1.0,
                'partial_match': 1.0,
                'coherence': 1.0,
                'relevance': 1.0,
                'exact_match': 1.0,
                'keywords_matched': [],
                'keywords_missed': [],
                'total_keywords': 0,
                'accuracy_percentage': 100.0,
                'combined_score': 1.0,
                'penalty': 0
            }
        }

    # Extract keywords using improved method
    keywords = extract_keywords_improved(expected_answer, student_answer)

    # Match keywords with flexible matching
    keywords_matched, keywords_missed = match_keywords_flexible(keywords, student_answer)

    # Calculate various similarity metrics
    cosine_sim = cosine_similarity_score(expected, student)
    semantic_sim = semantic_similarity_score(expected, student)
    syntactic_sim = calculate_syntactic_similarity(expected_answer, student_answer)

    # Enhanced sentence matching (combines semantic and syntactic)
    enhanced_match = enhanced_sentence_match(expected, student)

    # Calculate keyword coverage
    keyword_coverage = len(keywords_matched) / len(keywords) if keywords else 0

    # Calculate partial match (word overlap)
    partial_match_val = partial_match(expected, student)

    # Calculate coherence (length similarity)
    coherence_val = coherence_score(expected, student)

    # Calculate relevance (concept overlap)
    relevance_val = relevance_score(expected, student)

    # Exact match bonus
    exact_match_val = 1.0 if expected == student else 0.0

    # IMPROVED SCORING - Focus on Semantic Understanding with Fair Marking
    # Use a blend of semantic and cosine similarity as the primary score
    base_score = (semantic_sim * 0.70) + (cosine_sim * 0.30)
    
    # Generous boosts for any meaningful similarity
    if semantic_sim > 0.4:
        base_score = min(1.0, base_score * 1.15)  # 15% boost
    
    if semantic_sim > 0.6:
        base_score = min(1.0, base_score * 1.25)  # 25% boost for good similarity
    
    if semantic_sim > 0.75:
        base_score = min(1.0, base_score + 0.15)  # Add 15% for very good
    
    # Enhanced match bonus
    if enhanced_match > 0.5:
        base_score = min(1.0, base_score + 0.1)
    
    # Apply keyword coverage bonus (reward matching key concepts)
    if keyword_coverage > 0.5:
        base_score = min(1.0, base_score + (keyword_coverage * 0.1))
    
    # Apply relevance bonus
    if relevance_val > 0.4:
        base_score = min(1.0, base_score + (relevance_val * 0.05))
    
    # Ensure fair minimum scores based on similarity level
    # This ensures students are rewarded for partial understanding
    if base_score < 0.2 and (semantic_sim > 0.2 or cosine_sim > 0.2):
        base_score = 0.2  # Minimum 20% for some similarity
    elif base_score < 0.4 and (semantic_sim > 0.35 or cosine_sim > 0.35):
        base_score = 0.4  # 40% for moderate similarity
    elif base_score < 0.6 and (semantic_sim > 0.50 or cosine_sim > 0.50):
        base_score = min(0.6, base_score + 0.1)  # Boost to at least 60% for good similarity
    
    combined_score = base_score

    # Scale to total marks with decimal precision
    final_score = round(combined_score * total_marks, 2)

    # Ensure score is within bounds
    final_score = max(0.0, min(float(total_marks), final_score))

    # Calculate accuracy percentage
    accuracy_percentage = round(combined_score * 100, 2)

    return {
        'score': final_score,
        'details': {
            'cosine_similarity': round(cosine_sim, 4),
            'semantic_similarity': round(semantic_sim, 4),
            'syntactic_similarity': round(syntactic_sim, 4),
            'enhanced_sentence_match': round(enhanced_match, 4),
            'keyword_coverage': round(keyword_coverage, 4),
            'partial_match': round(partial_match_val, 4),
            'coherence': round(coherence_val, 4),
            'relevance': round(relevance_val, 4),
            'exact_match': exact_match_val,
            'keywords_matched': keywords_matched,
            'keywords_missed': keywords_missed,
            'total_keywords': len(keywords),
            'accuracy_percentage': accuracy_percentage,
            'combined_score': round(combined_score, 4),
            'penalty': 0
        }
    }

def evaluate(expected, response):
    """
    Legacy evaluate function - now uses improved evaluation with default total_marks
    """
    return evaluate_answers(expected, response, 10)

# Admin login route
@app.route('/')
def index():
    return render_template('Homepage.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM Admins WHERE username = %s AND password = %s", (username, password))
        admin = cur.fetchone()
        cur.close()

        if admin:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_home'))
        else:
            return render_template('adminlogin.html', error='Invalid username or password')

    return render_template('adminlogin.html')



# Admin home route
@app.route('/admin/home')
def admin_home():
    if 'admin_logged_in' in session:
        return render_template('adminhome.html')
    else:
        return redirect(url_for('admin_login'))

# Admin students route
@app.route('/admin/students')
def admin_students():
    if 'admin_logged_in' in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM Students")
        students = cur.fetchall()
        cur.close()
        return render_template('admin_students.html', students=students)
    else:
        return redirect(url_for('admin_login'))

# Add student route
@app.route('/admin/add_student', methods=['POST'])
def add_student():
    if 'admin_logged_in' in session:
        username = request.form['username']
        password = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO Students (username, password) VALUES (%s, %s)", (username, password))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('admin_students'))
    else:
        return redirect(url_for('admin_login'))

# Update student route
@app.route('/admin/update_student/<int:student_id>', methods=['POST'])
def update_student(student_id):
    if 'admin_logged_in' in session:
        username = request.form['username']
        password = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute("UPDATE Students SET username = %s, password = %s WHERE student_id = %s", (username, password, student_id))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('admin_students'))
    else:
        return redirect(url_for('admin_login'))

# Delete student route
@app.route('/admin/delete_student/<int:student_id>', methods=['POST'])
def delete_student(student_id):
    if 'admin_logged_in' in session:
        cur = mysql.connection.cursor()
        cur.execute("DELETE FROM Students WHERE student_id = %s", (student_id,))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('admin_students'))
    else:
        return redirect(url_for('admin_login'))

# View student scores route
# View student scores route
# View student scores route
@app.route('/admin/view_student_scores/<int:student_id>')
def view_student_scores(student_id):
    if 'admin_logged_in' in session:

        cur = mysql.connection.cursor()

        query = """
        SELECT answer_id, student_id, test_id, extracted_text, score
        FROM StudentAnswerSheets
        WHERE student_id = %s
        """

        cur.execute(query, (student_id,))
        data = cur.fetchall()

        cur.close()

        scores = [{
            'answer_id': row[0],
            'student_id': row[1],
            'test_id': row[2],
            'student_answer': row[3],
            'score': row[4]
        } for row in data]

        return render_template('student_scores.html', scores=scores)

    else:
        return redirect(url_for('admin_login'))
        
@app.route('/admin/delete_student_score/<int:answer_id>', methods=['POST'])
def delete_student_score(answer_id):
    if 'admin_logged_in' in session:
        cur = mysql.connection.cursor()
        # Delete the score with the given answer_id
        query = "DELETE FROM studentanswers WHERE answer_id = %s"
        cur.execute(query, (answer_id,))
        mysql.connection.commit()
        cur.close()
        return redirect(url_for('admin_students'))  # Redirect to admin students page
    else:
        return redirect(url_for('admin_login'))

###############################################################
#############################Admin Teacher ####################

# Admin teachers route
@app.route('/admin/teachers')
def admin_teachers():
    if 'admin_logged_in' in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM Teachers")
        teachers = cur.fetchall()
        cur.close()
        return render_template('admin_teachers.html', teachers=teachers)
    else:
        return redirect(url_for('admin_login'))

# Admin add teacher route
@app.route('/admin/add_teacher', methods=['GET', 'POST'])
def add_teacher():
    if 'admin_logged_in' in session:
        if request.method == 'POST':
            username = request.form['username']  # Changed from 'name' to 'username'
            password = request.form['password']  # Changed from 'email' to 'password'
            cur = mysql.connection.cursor()
            cur.execute("INSERT INTO Teachers (username, password) VALUES (%s, %s)", (username, password))
            mysql.connection.commit()
            cur.close()
            return redirect(url_for('admin_teachers'))
        else:
            return render_template('add_teacher.html')
    else:
        return redirect(url_for('admin_login'))

@app.route('/admin/update_teacher/<int:teacher_id>', methods=['GET', 'POST'])
def update_teacher(teacher_id):
    if 'admin_logged_in' in session:
        if request.method == 'POST':
            try:
                username = request.form['username']
                password = request.form['password']
                cur = mysql.connection.cursor()
                cur.execute("UPDATE Teachers SET username = %s, password = %s WHERE teacher_id = %s", (username, password, teacher_id))
                mysql.connection.commit()
                cur.close()
                return redirect(url_for('admin_teachers'))
            except Exception as e:
                print("Error updating teacher:", e)
                # Handle error appropriately, such as displaying an error message to the user
        else:
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM Teachers WHERE teacher_id = %s", (teacher_id,))
            teacher = cur.fetchone()
            cur.close()
            if teacher:
                return render_template('update_teacher.html', teacher=teacher, teacher_id=teacher_id)
            else:
                # Handle case where teacher with given ID is not found
                return "Teacher not found"
    else:
        return redirect(url_for('admin_login'))

# Admin delete teacher route
@app.route('/admin/delete_teacher/<int:teacher_id>', methods=['POST'])
def delete_teacher(teacher_id):
    if 'admin_logged_in' in session:
        try:
            cur = mysql.connection.cursor()

            # Delete related records from teacherstudentrelationship table
            cur.execute("DELETE FROM teacherstudentrelationship WHERE teacher_id = %s", (teacher_id,))
            
            # Now, delete the teacher from the teachers table
            cur.execute("DELETE FROM teachers WHERE teacher_id = %s", (teacher_id,))
            
            mysql.connection.commit()
            cur.close()
            return redirect(url_for('admin_teachers'))
        except Exception as e:
            # Handle any exceptions
            flash("An error occurred while deleting the teacher.")
            print(e)  # Print the exception for debugging purposes
            return redirect(url_for('admin_teachers'))
    else:
        return redirect(url_for('admin_login'))


# Admin view teacher tests route
@app.route('/admin/view_teacher_tests/<int:teacher_id>')
def view_teacher_tests(teacher_id):
    if 'admin_logged_in' in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM Tests WHERE teacher_id = %s", (teacher_id,))
        tests = cur.fetchall()
        cur.close()
        return render_template('view_teacher_tests.html', tests=tests, teacher_id=teacher_id)
    else:
        return redirect(url_for('admin_login'))

# Admin view test questions route
@app.route('/admin/view_test_questions/<int:test_id>')
def view_test_questions(test_id):
    if 'admin_logged_in' in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM Questions WHERE test_id = %s", (test_id,))
        questions = cur.fetchall()

        # Fetching expected answers for each question
        question_answers = {}
        for question in questions:
            cur.execute("SELECT * FROM ExpectedAnswers WHERE question_id = %s", (question[0],))
            answers = cur.fetchall()
            question_answers[question[0]] = answers

        cur.close()
        # Pass test_id as teacher_id to the template
        return render_template('view_test_questions.html', teacher_id=test_id, questions=questions, question_answers=question_answers)
    else:
        return redirect(url_for('admin_login'))

# Admin view question expected answers route
@app.route('/admin/view_question_answers/<int:question_id>')
def view_question_answers(question_id):
    if 'admin_logged_in' in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM ExpectedAnswers WHERE question_id = %s", (question_id,))
        answers = cur.fetchall()
        cur.close()
        return render_template('view_question_answers.html', answers=answers)
    else:
        return redirect(url_for('admin_login'))


################################################
    


# Admin logout route
@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))




################################################################################
###################################Teacher LOGIN######################
# Teacher login route
@app.route('/teacher_login', methods=['GET', 'POST'])
def teacher_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM Teachers WHERE username = %s AND password = %s", (username, password))
        teacher = cur.fetchone()
        cur.close()

        if teacher:
            session['teacher_logged_in'] = True
            session['teacher_id'] = teacher[0]  # Assuming teacher_id is the first column
            return redirect(url_for('teacher_home'))
        else:
            return render_template('teacher_login.html', error='Invalid username or password')

    return render_template('teacher_login.html')

# Teacher home route
@app.route('/teacher_home', methods=['GET', 'POST'])
def teacher_home():
    if 'teacher_logged_in' in session:
        if request.method == 'POST':
            # Check if form was submitted for adding, updating, or deleting test name
            if 'add_test_name' in request.form:
                test_name = request.form['test_name']
                total_marks = request.form['total_marks']
                # Add test name to the database
                cur = mysql.connection.cursor()
                cur.execute("INSERT INTO Tests (test_name, teacher_id, total_marks) VALUES (%s, %s, %s)", (test_name, session['teacher_id'], total_marks))
                mysql.connection.commit()
                cur.close()
            elif 'update_test_name' in request.form:
                test_id = request.form['test_id']
                updated_test_name = request.form['updated_test_name']
                # Update test name in the database
                cur = mysql.connection.cursor()
                cur.execute("UPDATE Tests SET test_name = %s WHERE test_id = %s", (updated_test_name, test_id))
                mysql.connection.commit()
                cur.close()
            elif 'delete_test_name' in request.form:
                test_id = request.form['test_id']
                
                try:
                    # Delete related student answers first
                    cur = mysql.connection.cursor()
                    cur.execute("DELETE FROM studentanswers WHERE test_id = %s", (test_id,))
                    mysql.connection.commit()
                    cur.close()

                    # Delete related expected answers
                    cur = mysql.connection.cursor()
                    cur.execute("DELETE FROM expectedanswers WHERE question_id IN (SELECT question_id FROM questions WHERE test_id = %s)", (test_id,))
                    mysql.connection.commit()
                    cur.close()

                    # Delete related questions
                    cur = mysql.connection.cursor()
                    cur.execute("DELETE FROM questions WHERE test_id = %s", (test_id,))
                    mysql.connection.commit()
                    cur.close()

                    # Now delete the test from the Tests table
                    cur = mysql.connection.cursor()
                    cur.execute("DELETE FROM tests WHERE test_id = %s", (test_id,))
                    mysql.connection.commit()
                    cur.close()

                except Exception as e:
                    # Handle any exceptions
                    print("Error:", e)





        # Fetch all tests for the current teacher
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM Tests WHERE teacher_id = %s", (session['teacher_id'],))
        tests = cur.fetchall()
        cur.close()
        return render_template('teacher_home.html', tests=tests)
    else:
        return redirect(url_for('teacher_login'))

# Teacher logout route
@app.route('/teacher_logout')
def teacher_logout():
    session.pop('teacher_logged_in', None)
    session.pop('teacher_id', None)
    return redirect(url_for('teacher_login'))

# Teacher Test Panel route
@app.route('/teacher_test_panel')
def teacher_test_panel():
    if 'teacher_logged_in' in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM Tests WHERE teacher_id = %s", (session['teacher_id'],))
        tests = cur.fetchall()
        cur.close()
        return render_template('teacher_test_panel.html', tests=tests)
    else:
        return redirect(url_for('teacher_login'))

######################teacherLOGOUT####################################
################################################################################
###############teacher FUNCTIONS ############################
@app.route('/teacher/view_test_questions/<int:test_id>', methods=['GET', 'POST'])
def view_teacher_test_questions(test_id):
    if 'teacher_logged_in' in session:
        if request.method == 'POST':
            if 'add_question' in request.form:
                question_text = request.form['question_text']
                expected_answers = request.form.getlist('expected_answer')
                cur = mysql.connection.cursor()
                cur.execute("INSERT INTO Questions (question_text, test_id) VALUES (%s, %s)", (question_text, test_id))
                question_id = cur.lastrowid
                for answer in expected_answers:
                    cur.execute("INSERT INTO ExpectedAnswers (answer_text, question_id) VALUES (%s, %s)", (answer, question_id))
                mysql.connection.commit()
                cur.close()
            elif 'delete_question' in request.form:
                question_id = request.form['question_id']
                cur = mysql.connection.cursor()
                cur.execute("DELETE FROM ExpectedAnswers WHERE question_id = %s", (question_id,))
                cur.execute("DELETE FROM Questions WHERE question_id = %s", (question_id,))
                mysql.connection.commit()
                cur.close()

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM Questions WHERE test_id = %s", (test_id,))
        questions = cur.fetchall()

        question_answers = {}
        for question in questions:
            cur.execute("SELECT * FROM ExpectedAnswers WHERE question_id = %s", (question[0],))
            answers = cur.fetchall()
            question_answers[question[0]] = answers

        cur.close()
        return render_template('view_teacher_test_questions.html', test_id=test_id, questions=questions, question_answers=question_answers)
    else:
        return redirect(url_for('teacher_login'))

###### Teacher ( student marks section page) ################
@app.route('/teacher_view_score')
def teacher_view_score():
    # Check if the user is logged in as a teacher
    if 'teacher_logged_in' in session:
        teacher_id = session['teacher_id']

        # Fetch student answers and expected answers for the logged-in teacher's tests
        cur = mysql.connection.cursor()
        query = """
            SELECT s.student_id, s.username AS student_username, t.test_name, q.question_text, ea.answer_text AS expected_answer, sa.answer_text AS student_answer, sa.score, sa.evaluation_details
            FROM StudentAnswers sa
            JOIN Students s ON sa.student_id = s.student_id
            JOIN Tests t ON sa.test_id = t.test_id
            JOIN Questions q ON sa.question_id = q.question_id
            JOIN ExpectedAnswers ea ON q.question_id = ea.question_id
            WHERE t.teacher_id = %s
        """
        cur.execute(query, (teacher_id,))
        results = cur.fetchall()

        # Group the results by student_id and test_name
        student_scores = defaultdict(lambda: {'student_username': None, 'tests': defaultdict(list)})
        for result in results:
            student_id, student_username, test_name, question_text, expected_answer, student_answer, score, evaluation_details = result
            details = json.loads(evaluation_details) if evaluation_details else None
            student_scores[student_id]['student_username'] = student_username
            student_scores[student_id]['tests'][test_name].append({
                'question_text': question_text,
                'expected_answer': expected_answer,
                'student_answer': student_answer,
                'score': score,
                'details': details
            })

        return render_template('teacher_view_score.html', student_scores=student_scores)
    else:
        return redirect(url_for('teacher_login'))


# Route for teachers to view uploaded answer sheets for their tests
@app.route('/teacher/view_uploaded_sheets')
def teacher_view_uploaded_sheets():
    if 'teacher_logged_in' in session:
        teacher_id = session['teacher_id']
        cur = mysql.connection.cursor()
        query = """
            SELECT sas.sheet_id, s.username AS student_username, t.test_name, sas.score, sas.uploaded_at, sas.evaluation_details
            FROM StudentAnswerSheets sas
            JOIN Students s ON sas.student_id = s.student_id
            JOIN Tests t ON sas.test_id = t.test_id
            WHERE t.teacher_id = %s
            ORDER BY sas.uploaded_at DESC
        """
        cur.execute(query, (teacher_id,))
        sheets_data = cur.fetchall()
        cur.close()
        
        # Process the data to include parsed evaluation details safely
        sheets = []
        for sheet in sheets_data:
            sheet_id, student_username, test_name, score, uploaded_at, evaluation_details = sheet
            details = {}
            if evaluation_details:
                try:
                    details = json.loads(evaluation_details)
                except Exception as e:
                    print(f"Invalid evaluation_details JSON for sheet_id {sheet_id}: {e}")
                    details = {'error': 'Invalid evaluation details'}

            # Ensure required keys are always present with new structure
            defaults = {
                'cosine_similarity': 0,
                'semantic_similarity': 0,
                'syntactic_similarity': 0,
                'enhanced_sentence_match': 0,
                'keyword_coverage': 0,
                'partial_match': 0,
                'coherence': 0,
                'relevance': 0,
                'exact_match': 0,
                'keywords_matched': [],
                'keywords_missed': [],
                'total_keywords': 0,
                'accuracy_percentage': 0,
                'combined_score': 0,
                'penalty': 0
            }
            for key, val in defaults.items():
                details.setdefault(key, val)

            sheets.append({
                'sheet_id': sheet_id,
                'student_username': student_username,
                'test_name': test_name,
                'score': score,
                'uploaded_at': uploaded_at,
                'details': details
            })

        return render_template('teacher_view_uploaded_sheets.html', sheets=sheets)
    else:
        return redirect(url_for('teacher_login'))


# Route to view detailed evaluation for a specific sheet
@app.route('/teacher/sheet_details/<int:sheet_id>')
def view_sheet_details(sheet_id):
    if 'teacher_logged_in' in session:
        teacher_id = session['teacher_id']
        cur = mysql.connection.cursor()
        
        # Get sheet details with model answer key
        query = """
            SELECT sas.sheet_id, s.username AS student_username, t.test_name, t.total_marks, 
                   sas.score, sas.uploaded_at, sas.evaluation_details, sas.extracted_text, mak.file_path
            FROM StudentAnswerSheets sas
            JOIN Students s ON sas.student_id = s.student_id
            JOIN Tests t ON sas.test_id = t.test_id
            LEFT JOIN ModelAnswerKeys mak ON t.test_id = mak.test_id
            WHERE sas.sheet_id = %s AND t.teacher_id = %s
            ORDER BY mak.uploaded_at DESC LIMIT 1
        """
        cur.execute(query, (sheet_id, teacher_id))
        sheet = cur.fetchone()
        cur.close()
        
        if not sheet:
            flash('Sheet not found or access denied')
            return redirect(url_for('teacher_view_uploaded_sheets'))
        
        sheet_id, student_username, test_name, total_marks, score, uploaded_at, evaluation_details, extracted_text, model_key_path = sheet
        
        # Parse evaluation details
        details = {}
        if evaluation_details:
            try:
                details = json.loads(evaluation_details)
            except Exception as e:
                print(f"Invalid evaluation_details JSON: {e}")
                details = {'error': 'Invalid evaluation details'}
        
        # Ensure defaults with new structure
        defaults = {
            'cosine_similarity': 0,
            'semantic_similarity': 0,
            'syntactic_similarity': 0,
            'enhanced_sentence_match': 0,
            'keyword_coverage': 0,
            'partial_match': 0,
            'coherence': 0,
            'relevance': 0,
            'exact_match': 0,
            'keywords_matched': [],
            'keywords_missed': [],
            'total_keywords': 0,
            'accuracy_percentage': 0,
            'combined_score': 0,
            'penalty': 0
        }
        for key, val in defaults.items():
            details.setdefault(key, val)
        
        # Extract model answer
        model_answer = ""
        if model_key_path:
            try:
                model_answer = extract_text_from_file(model_key_path)
            except Exception as e:
                print(f"Error extracting model answer: {e}")
                model_answer = "Could not extract model answer"
        
        # Calculate additional metrics for display
        cosine_sim = details.get('cosine_similarity', 0)
        semantic_sim = details.get('semantic_similarity', 0)
        
        # Format percentages
        cosine_sim_percent = round(cosine_sim * 100, 2)
        semantic_sim_percent = round(semantic_sim * 100, 2)
        
        # Calculate feedback
        feedback = generate_feedback(score, total_marks, details, cosine_sim, semantic_sim)
        
        sheet_data = {
            'sheet_id': sheet_id,
            'student_username': student_username,
            'test_name': test_name,
            'total_marks': total_marks,
            'score': score,
            'uploaded_at': uploaded_at,
            'evaluation_details': details,
            'model_answer': model_answer,
            'student_answer': extracted_text,
            'cosine_similarity': cosine_sim_percent,
            'semantic_similarity': semantic_sim_percent,
            'feedback': feedback
        }
        
        return render_template('sheet_details.html', sheet=sheet_data)
    else:
        return redirect(url_for('teacher_login'))


def generate_feedback(score, total_marks, details, cosine_sim, semantic_sim):
    """Generate constructive feedback based on evaluation metrics"""
    feedback = []

    percentage = (score / total_marks * 100) if total_marks > 0 else 0

    # Overall performance feedback based on score out of total marks
    if percentage >= 95:
        feedback.append("🎉 Outstanding! Your answer is excellent and demonstrates complete mastery of the topic.")
        feedback.append(f"You scored {score}/{total_marks} ({percentage:.1f}%) - Keep up the excellent work!")
    elif percentage >= 85:
        feedback.append("🌟 Excellent work! Your answer shows strong understanding with minor areas for improvement.")
        feedback.append(f"You scored {score}/{total_marks} ({percentage:.1f}%) - Very well done!")
    elif percentage >= 75:
        feedback.append("👍 Good effort! Your answer covers most key points but could benefit from more detail.")
        feedback.append(f"You scored {score}/{total_marks} ({percentage:.1f}%) - Solid performance with room for enhancement.")
    elif percentage >= 65:
        feedback.append("📝 Fair attempt. Your answer has potential but needs more comprehensive coverage.")
        feedback.append(f"You scored {score}/{total_marks} ({percentage:.1f}%) - Focus on including more key concepts.")
    elif percentage >= 50:
        feedback.append("⚠️ Needs improvement. Your answer misses several important points.")
        feedback.append(f"You scored {score}/{total_marks} ({percentage:.1f}%) - Review the model answer and try again.")
    else:
        feedback.append("❌ Significant improvement needed. Your answer doesn't align well with the expected response.")
        feedback.append(f"You scored {score}/{total_marks} ({percentage:.1f}%) - Please study the topic thoroughly and retry.")

    feedback.append("")  # Add spacing

    # Detailed analysis based on metrics
    feedback.append("📊 Detailed Analysis:")

    # Cosine similarity feedback
    if cosine_sim < 0.3:
        feedback.append("• Word choice and structure differ significantly from the model answer.")
        feedback.append("  💡 Improvement: Study the model answer's vocabulary and sentence structure.")
    elif cosine_sim < 0.6:
        feedback.append("• Partial alignment in word choice. Some key terms are missing or differently phrased.")
        feedback.append("  💡 Improvement: Include more technical terms and subject-specific vocabulary.")
    else:
        feedback.append("• Good alignment with model answer in word choice and structure.")
        feedback.append("  ✅ Strength: Effective use of appropriate terminology.")

    # Semantic similarity feedback
    if semantic_sim < 0.3:
        feedback.append("• The core meaning and concepts differ substantially from the expected answer.")
        feedback.append("  💡 Improvement: Focus on understanding and explaining the main concepts clearly.")
    elif semantic_sim < 0.6:
        feedback.append("• Moderate semantic alignment. The meaning is partially conveyed but could be clearer.")
        feedback.append("  💡 Improvement: Work on expressing ideas more precisely and comprehensively.")
    else:
        feedback.append("• Strong semantic alignment with the model answer.")
        feedback.append("  ✅ Strength: Clear and accurate expression of concepts.")

    # Syntactic similarity feedback
    syntactic_sim = details.get('syntactic_similarity', 0)
    if syntactic_sim < 0.3:
        feedback.append("• Sentence structure and grammar need significant improvement.")
        feedback.append("  💡 Improvement: Practice proper sentence construction and grammar rules.")
    elif syntactic_sim < 0.6:
        feedback.append("• Sentence structure is developing but could be more sophisticated.")
        feedback.append("  💡 Improvement: Use varied sentence structures and complex grammatical forms.")
    else:
        feedback.append("• Good sentence structure and grammatical accuracy.")
        feedback.append("  ✅ Strength: Well-structured and grammatically correct writing.")

    # Keywords analysis
    keywords_matched = details.get('keywords_matched', [])
    keywords_missed = details.get('keywords_missed', [])
    keyword_coverage = details.get('keyword_coverage', 0)

    if keyword_coverage < 0.4:
        feedback.append(f"• Low keyword coverage ({keyword_coverage:.1f}). Many important terms are missing.")
        if keywords_missed:
            missed_sample = keywords_missed[:3]  # Show first 3 missed keywords
            feedback.append(f"  💡 Missing key terms: {', '.join(missed_sample)}")
        feedback.append("  💡 Improvement: Include more subject-specific terminology and key concepts.")
    elif keyword_coverage < 0.7:
        feedback.append(f"• Moderate keyword coverage ({keyword_coverage:.1f}). Some important terms are missing.")
        if keywords_missed:
            missed_sample = keywords_missed[:2]
            feedback.append(f"  💡 Consider adding: {', '.join(missed_sample)}")
        feedback.append("  💡 Improvement: Review the topic to identify and include all key terms.")
    else:
        feedback.append(f"• Excellent keyword coverage ({keyword_coverage:.1f}). Most important terms included.")
        feedback.append("  ✅ Strength: Comprehensive use of relevant terminology.")

    feedback.append("")  # Add spacing

    # Specific improvement recommendations based on score range
    feedback.append("🎯 Specific Recommendations to Improve Your Score:")

    if percentage < 70:
        feedback.append("• Study the model answer carefully and identify what key points you missed.")
        feedback.append("• Practice writing answers that include all main concepts and supporting details.")
        feedback.append("• Focus on using precise technical vocabulary related to the subject.")
        feedback.append("• Work on organizing your thoughts logically and expressing them clearly.")
    elif percentage < 85:
        feedback.append("• Add more depth and detail to your explanations.")
        feedback.append("• Ensure you've covered all aspects of the question comprehensively.")
        feedback.append("• Use more varied and sophisticated vocabulary.")
        feedback.append("• Review grammar and sentence structure for clarity.")
    else:
        feedback.append("• Continue refining your writing style and depth of analysis.")
        feedback.append("• Challenge yourself with more complex questions to further develop skills.")
        feedback.append("• Help peers by explaining concepts - teaching reinforces learning.")

    # Penalty information
    penalty = details.get('penalty', 0)
    if penalty > 0:
        feedback.append("")
        feedback.append(f"📝 Note: A penalty of {penalty} points was applied. Review the evaluation criteria and try to avoid similar issues in future submissions.")

    return feedback


def generate_pdf_report(sheet_data):
    """Generate PDF report for evaluation results"""
    if not HAS_REPORTLAB:
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=0.5*inch, leftMargin=0.5*inch,
                           topMargin=0.75*inch, bottomMargin=0.75*inch)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    story.append(Paragraph("Answer Evaluation Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Student and Test Info
    info_data = [
        ['Student Name:', sheet_data['student_username']],
        ['Test Name:', sheet_data['test_name']],
        ['Uploaded Date:', str(sheet_data['uploaded_at'])],
    ]
    
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0'))
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Score Section
    score_title = ParagraphStyle('ScoreTitle', parent=styles['Heading2'], fontSize=14, 
                                 textColor=colors.HexColor('#0f172a'), spaceAfter=12)
    story.append(Paragraph("Score", score_title))
    
    score_text = f"<b>Score: {sheet_data['score']}/{sheet_data['total_marks']} ({round(sheet_data['score']/sheet_data['total_marks']*100, 2)}%)</b>"
    score_style = ParagraphStyle('ScoreResult', parent=styles['Normal'], fontSize=14,
                                textColor=colors.HexColor('#10b981'), spaceAfter=20)
    story.append(Paragraph(score_text, score_style))
    
    # Similarity Metrics
    story.append(Paragraph("Similarity Analysis", score_title))
    
    metrics_data = [
        ['Metric', 'Score'],
        ['Cosine Similarity', f"{sheet_data['cosine_similarity']}%"],
        ['Semantic Similarity', f"{sheet_data['semantic_similarity']}%"],
    ]
    
    metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Feedback Section - Enhanced
    story.append(Paragraph("📝 Feedback & Recommendations", score_title))
    story.append(Spacer(1, 0.1*inch))
    
    # Overall performance summary
    performance_style = ParagraphStyle('Performance', parent=styles['Normal'], fontSize=11,
                                     textColor=colors.HexColor('#0f172a'), spaceAfter=15,
                                     fontName='Helvetica-Bold')
    
    percentage = (sheet_data['score'] / sheet_data['total_marks'] * 100) if sheet_data['total_marks'] > 0 else 0
    performance_text = f"Overall Performance: {sheet_data['score']}/{sheet_data['total_marks']} ({percentage:.1f}%)"
    story.append(Paragraph(performance_text, performance_style))
    
    feedback_style = ParagraphStyle('Feedback', parent=styles['Normal'], fontSize=9,
                                   leading=12, leftIndent=15, spaceAfter=6,
                                   textColor=colors.HexColor('#334155'))
    
    improvement_style = ParagraphStyle('Improvement', parent=styles['Normal'], fontSize=8,
                                     leading=10, leftIndent=25, spaceAfter=4,
                                     textColor=colors.HexColor('#64748b'))
    
    for feedback_item in sheet_data['feedback']:
        if feedback_item.strip():  # Skip empty lines
            if "💡" in feedback_item or "✅" in feedback_item or "🎯" in feedback_item:
                story.append(Paragraph(feedback_item, improvement_style))
            elif "📊" in feedback_item or "🎉" in feedback_item or "🌟" in feedback_item or "👍" in feedback_item or "📝" in feedback_item or "⚠️" in feedback_item or "❌" in feedback_item:
                story.append(Paragraph(feedback_item, feedback_style))
            else:
                story.append(Paragraph(feedback_item, feedback_style))
    
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Answers Section
    story.append(PageBreak())
    story.append(Paragraph("Detailed Comparison", score_title))
    story.append(Spacer(1, 0.2*inch))
    
    # Model Answer
    story.append(Paragraph("Model Answer (Expected):", 
                          ParagraphStyle('SubHead', parent=styles['Heading3'], fontSize=11, 
                                       textColor=colors.HexColor('#0f172a'))))
    
    def clean_paragraph_text(text):
        if not text:
            return ""
        safe_text = xml_escape(text, {'\n': '<br/>'})
        return safe_text.replace('\r', '')

    model_answer_text = sheet_data['model_answer'][:500] + "..." if len(sheet_data['model_answer']) > 500 else sheet_data['model_answer']
    story.append(Paragraph(clean_paragraph_text(model_answer_text), 
                          ParagraphStyle('AnswerText', parent=styles['Normal'], fontSize=9,
                                       leftIndent=20, rightIndent=20, spaceAfter=15,
                                       textColor=colors.HexColor('#334155'))))
    
    # Student Answer
    story.append(Paragraph("Student Answer (Provided):", 
                          ParagraphStyle('SubHead', parent=styles['Heading3'], fontSize=11,
                                       textColor=colors.HexColor('#0f172a'))))
    
    student_answer_text = sheet_data['student_answer'][:500] + "..." if len(sheet_data['student_answer']) > 500 else sheet_data['student_answer']
    story.append(Paragraph(clean_paragraph_text(student_answer_text),
                          ParagraphStyle('AnswerText', parent=styles['Normal'], fontSize=9,
                                       leftIndent=20, rightIndent=20, spaceAfter=15,
                                       textColor=colors.HexColor('#334155'))))
    
    # Keywords Analysis
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Keywords Analysis", score_title))
    
    keywords_data = [
        ['Matched Keywords', 'Missed Keywords'],
    ]
    matched = ', '.join(sheet_data['evaluation_details'].get('keywords_matched', [])[:5]) or 'None'
    missed = ', '.join(sheet_data['evaluation_details'].get('keywords_missed', [])[:5]) or 'None'
    keywords_data.append([matched, missed])
    
    keywords_table = Table(keywords_data, colWidths=[3*inch, 3*inch])
    keywords_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    story.append(keywords_table)
    
    # Footer
    story.append(Spacer(1, 0.4*inch))
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8,
                                 textColor=colors.HexColor('#94a3b8'), alignment=TA_CENTER)
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | AES_ai Evaluation System",
                          footer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


# Route to download evaluation report as PDF
@app.route('/teacher/sheet_download_report/<int:sheet_id>')
def download_sheet_report(sheet_id):
    if 'teacher_logged_in' in session:
        if not HAS_REPORTLAB:
            flash('PDF generation is not available. Please install reportlab: pip install reportlab')
            return redirect(url_for('view_sheet_details', sheet_id=sheet_id))
        
        teacher_id = session['teacher_id']
        cur = mysql.connection.cursor()
        
        # Get sheet details
        query = """
            SELECT sas.sheet_id, s.username AS student_username, t.test_name, t.total_marks, 
                   sas.score, sas.uploaded_at, sas.evaluation_details, sas.extracted_text, mak.file_path
            FROM StudentAnswerSheets sas
            JOIN Students s ON sas.student_id = s.student_id
            JOIN Tests t ON sas.test_id = t.test_id
            LEFT JOIN ModelAnswerKeys mak ON t.test_id = mak.test_id
            WHERE sas.sheet_id = %s AND t.teacher_id = %s
            ORDER BY mak.uploaded_at DESC LIMIT 1
        """
        cur.execute(query, (sheet_id, teacher_id))
        sheet = cur.fetchone()
        cur.close()
        
        if not sheet:
            flash('Sheet not found or access denied')
            return redirect(url_for('teacher_view_uploaded_sheets'))
        
        sheet_id, student_username, test_name, total_marks, score, uploaded_at, evaluation_details, extracted_text, model_key_path = sheet
        
        # Parse evaluation details
        details = {}
        if evaluation_details:
            try:
                details = json.loads(evaluation_details)
            except:
                details = {}
        
        # Extract model answer
        model_answer = ""
        if model_key_path:
            try:
                model_answer = extract_text_from_file(model_key_path)
            except:
                model_answer = "Could not extract model answer"
        
        # Calculate metrics
        cosine_sim = details.get('cosine_similarity', 0)
        semantic_sim = details.get('semantic_similarity', 0)
        cosine_sim_percent = round(cosine_sim * 100, 2)
        semantic_sim_percent = round(semantic_sim * 100, 2)
        
        # Generate feedback
        feedback = generate_feedback(score, total_marks, details, cosine_sim, semantic_sim)
        
        sheet_data = {
            'sheet_id': sheet_id,
            'student_username': student_username,
            'test_name': test_name,
            'total_marks': total_marks,
            'score': score,
            'uploaded_at': uploaded_at,
            'evaluation_details': details,
            'model_answer': model_answer,
            'student_answer': extracted_text,
            'cosine_similarity': cosine_sim_percent,
            'semantic_similarity': semantic_sim_percent,
            'feedback': feedback
        }
        
        # Generate PDF
        pdf_buffer = generate_pdf_report(sheet_data)
        
        if pdf_buffer is None:
            flash('Error generating PDF report')
            return redirect(url_for('view_sheet_details', sheet_id=sheet_id))
        
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'Evaluation_Report_{student_username}_{test_name}_{datetime.now().strftime("%Y%m%d")}.pdf'
        )
    else:
        return redirect(url_for('teacher_login'))
                                                              
######################## Student LOGIN ####################### 
    
@app.route('/student_login', methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM Students WHERE username = %s AND password = %s", (username, password))
        student = cur.fetchone()
        cur.close()

        if student:
            session['student_logged_in'] = True
            session['student_id'] = student[0]  # Assuming student_id is the first column
            return redirect(url_for('student_home'))
        else:
            return render_template('student_login.html', error='Invalid username or password')

    return render_template('student_login.html')
@app.route('/student_home')
def student_home():
    if 'student_logged_in' in session:
        return render_template('student_home.html')
    else:
        return redirect(url_for('student_login'))

@app.route('/student_logout')
def student_logout():
    session.pop('student_logged_in', None)
    session.pop('student_id', None)
    return redirect(url_for('student_login'))



# Route for showing available tests and taking a test
@app.route('/student_take_test', methods=['GET', 'POST'])
def student_take_test():
    if 'student_logged_in' in session:
        if request.method == 'POST':
            # Handle form submission (store student answers)
            test_id = request.form.get('test_id')  # Assuming you have a hidden input field for the test_id
            student_id = session['student_id']  # Assuming you have stored student_id in the session
            
            # Check if the student has already taken the test
            if check_test_taken(student_id):
                return redirect(url_for('student_view_score'))
            
            # Loop through form data to retrieve answers for each question
            for question_id, answer in request.form.items():
                # Assuming input field names are in the format 'question_{question_id}'
                if question_id.startswith('question_'):
                    question_id = int(question_id.split('_')[1])
                    
                    # Store student answer in the StudentAnswers table
                    cur = mysql.connection.cursor()
                    cur.execute("INSERT INTO StudentAnswers (student_id, test_id, question_id, answer_text) VALUES (%s, %s, %s, %s)",
                                (student_id, test_id, question_id, answer))
                    mysql.connection.commit()
                    cur.close()
            
            # Redirect the student after storing answers
            return redirect(url_for('student_view_score'))
        else:
            # Fetch tests that the student has not taken yet
            cur = mysql.connection.cursor()
            cur.execute("""SELECT t.test_id, t.test_name 
                           FROM Tests t 
                           LEFT JOIN StudentAnswers sa ON t.test_id = sa.test_id AND sa.student_id = %s
                           WHERE sa.test_id IS NULL""", (session['student_id'],))
            tests = cur.fetchall()
            cur.close()
            
            # Convert the list of tuples to a list of dictionaries
            tests = [{'test_id': test[0], 'test_name': test[1]} for test in tests]
            
            return render_template('student_take_test.html', tests=tests)
    else:
        return redirect(url_for('student_login'))

@app.route('/student_take_test/<int:test_id>', methods=['GET', 'POST'])
def student_take_test_questions(test_id):
    if 'student_logged_in' in session:
        if request.method == 'POST':
            # Retrieve student ID from the session
            student_id = session['student_id']
            
            # Retrieve test ID from the route parameter
            test_id = test_id
            
            # Get test details to calculate marks per question
            cur = mysql.connection.cursor()
            cur.execute("SELECT total_marks FROM Tests WHERE test_id = %s", (test_id,))
            test_data = cur.fetchone()
            total_marks_for_test = test_data[0] if test_data else 10
            
            # Get the number of questions in this test
            cur.execute("SELECT COUNT(*) FROM Questions WHERE test_id = %s", (test_id,))
            num_questions = cur.fetchone()[0]
            marks_per_question = total_marks_for_test / num_questions if num_questions > 0 else float(total_marks_for_test)
            
            # Loop through form data to retrieve answers for each question
            for question_id, answer in request.form.items():
                # Assuming input field names are in the format 'question_{question_id}'
                if question_id.startswith('question_'):
                    question_id = int(question_id.split('_')[1])

                    # Fetch the expected answer for this question
                    cur.execute("SELECT answer_text FROM ExpectedAnswers WHERE question_id = %s LIMIT 1", (question_id,))
                    expected_answer_row = cur.fetchone()
                    expected_answer = expected_answer_row[0] if expected_answer_row else ""

                    # Evaluate the student answer
                    evaluation_result = evaluate_answers(expected_answer, answer, marks_per_question)
                    score = round(float(evaluation_result['score']), 2)
                    evaluation_details = json.dumps(evaluation_result['details'])

                    # Store student answer with score and evaluation details in the StudentAnswers table
                    cur.execute("INSERT INTO StudentAnswers (student_id, test_id, question_id, answer_text, score, evaluation_details) VALUES (%s, %s, %s, %s, %s, %s)",
                                (student_id, test_id, question_id, answer, score, evaluation_details))
                    mysql.connection.commit()

            cur.close()

            # Redirect the student after storing answers
            return redirect(url_for('student_home'))

        else:
            # Fetch test details and questions for the specified test from the database
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM Tests WHERE test_id = %s", (test_id,))
            test = cur.fetchone()
            cur.execute("SELECT * FROM Questions WHERE test_id = %s", (test_id,))
            questions = cur.fetchall()
            cur.close()

            return render_template('student_take_test_questions.html', test=test, questions=questions, test_id=test_id)
    else:
        return redirect(url_for('student_login'))

@app.route('/student_view_score')
def student_view_score():
    # Check if the user is logged in as a student
    if 'student_logged_in' in session:
        student_id = session['student_id']

        # Fetch student answers, test names, questions, and expected answers for the logged-in student's tests
        cur = mysql.connection.cursor()
        query = """
            SELECT t.test_id, t.test_name, t.total_marks, q.question_text, ea.answer_text AS expected_answer, 
                   sa.answer_text AS student_answer, sa.score, sa.evaluation_details
            FROM StudentAnswers sa
            JOIN Tests t ON sa.test_id = t.test_id
            JOIN Questions q ON sa.question_id = q.question_id
            JOIN ExpectedAnswers ea ON q.question_id = ea.question_id
            WHERE sa.student_id = %s
            ORDER BY t.test_id
        """
        cur.execute(query, (student_id,))
        results = cur.fetchall()
        cur.close()

        # Prepare the data to be displayed
        student_scores = {}
        for result in results:
            test_id, test_name, total_marks, question_text, expected_answer, student_answer, score, evaluation_details = result
            details = json.loads(evaluation_details) if evaluation_details else None
            
            # Check if test_id already exists in student_scores
            if test_id not in student_scores:
                student_scores[test_id] = {
                    'test_id': test_id,
                    'test_name': test_name,
                    'total_marks': total_marks,
                    'total_score': 0,
                    'num_questions': 0,
                    'scores': []
                }
            
            student_scores[test_id]['scores'].append({
                'question': question_text,
                'expected_answer': expected_answer,
                'student_answer': student_answer,
                'score': score if score else 0,
                'details': details
            })
            
            # Accumulate total score
            student_scores[test_id]['total_score'] += (score if score else 0)
            student_scores[test_id]['num_questions'] += 1

        # Format total score for each test
        for test_data in student_scores.values():
            total_score = test_data['total_score']
            total_marks = test_data['total_marks']
            # Round to nearest integer and cap at total_marks
            total_score = round(min(total_score, total_marks))
            test_data['total_score'] = f"{total_score} / {total_marks}"

        return render_template('student_view_score.html', student_scores=student_scores.values())
    else:
        return redirect(url_for('student_login'))

@app.route('/student/view_uploaded_scores')
def view_uploaded_scores():
    if 'student_logged_in' in session:
        student_id = session['student_id']
        cur = mysql.connection.cursor()
        query = """
            SELECT t.test_name, sas.score, sas.uploaded_at, sas.evaluation_details
            FROM StudentAnswerSheets sas
            JOIN Tests t ON sas.test_id = t.test_id
            WHERE sas.student_id = %s
            ORDER BY sas.uploaded_at DESC
        """
        cur.execute(query, (student_id,))
        scores_data = cur.fetchall()
        cur.close()
        
        # Process the data to include parsed evaluation details
        scores = []
        for score_data in scores_data:
            test_name, score, uploaded_at, evaluation_details = score_data
            details = json.loads(evaluation_details) if evaluation_details else None
            scores.append({
                'test_name': test_name,
                'score': score,
                'uploaded_at': uploaded_at,
                'details': details
            })
        
        return render_template('student_view_uploaded_scores.html', scores=scores)
    else:
        return redirect(url_for('student_login'))


# Helper functions for file uploads
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_ocr(image_path):
    """
    Lightweight preprocessing for OCR (optimized for speed).
    Skips deskewing and complex filtering for faster processing.
    """
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Light denoising only
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Simple thresholding (faster than CLAHE)
    _, threshold = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY)
    
    # Save preprocessed image temporarily
    temp_path = image_path + '.processed.png'
    cv2.imwrite(temp_path, threshold)
    
    return temp_path

def ocr_with_multiple_psm(image_obj):
    """Try multiple PSM/OEM modes for robust OCR with fallback."""
    modes = [
        {'psm': 6, 'oem': 3},
        {'psm': 3, 'oem': 3},
        {'psm': 6, 'oem': 1},
        {'psm': 1, 'oem': 3}
    ]
    for mode in modes:
        try:
            text = pytesseract.image_to_string(
                image_obj,
                config=f"--psm {mode['psm']} -l eng --oem {mode['oem']}",
            )
            text = ' '.join(text.split())
            if len(text) > 10:
                return text
        except Exception as e:
            print(f"OCR attempt psm={mode['psm']} oem={mode['oem']} failed: {e}")
            continue
    return ''


def extract_text_from_pdf(file_path, dpi=150, max_pages=20):
    """
    Extract text from PDF with fallback to image-based OCR for scanned PDFs (optimized)
    """
    text = ""
    
    try:
        # First, try to extract text directly (for text-based PDFs)
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted
        
        # If we got significant text, return it
        if len(text.strip()) > 50:
            return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    
    # If direct extraction failed or got minimal text, try pdfplumber if available first
    if pdfplumber is not None:
        try:
            with pdfplumber.open(file_path) as pdf:
                pdfplumber_text = ' '.join([p.extract_text() or '' for p in pdf.pages])
                pdfplumber_text = ' '.join(pdfplumber_text.split())
                if len(pdfplumber_text) > 50:
                    return pdfplumber_text
        except Exception as e:
            print(f"pdfplumber extraction failed: {e}")

    # Next step: convert PDF to images and use OCR (OPTIMIZED: reduced DPI and page limit)
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert PDF pages to images at requested DPI
            # Limit pages to avoid excessive processing
            images = convert_from_path(file_path, dpi=dpi, output_folder=temp_dir, first_page=1, last_page=max_pages)
            
            for i, image in enumerate(images):
                # Save image temporarily
                img_path = os.path.join(temp_dir, f'page_{i}.png')
                image.save(img_path, 'PNG')

                # Try OCR directly first (faster than preprocessing)
                page_text = ocr_with_multiple_psm(Image.open(img_path))
                
                # If OCR result is poor, try with preprocessing
                if page_text and len(page_text) > 20:
                    text += page_text + "\n"
                else:
                    processed_path = preprocess_image_for_ocr(img_path)
                    if processed_path and os.path.exists(processed_path):
                        page_text = ocr_with_multiple_psm(Image.open(processed_path))
                        text += page_text + "\n"
                        # Clean up processed image
                        if os.path.exists(processed_path):
                            os.remove(processed_path)
                    else:
                        text += page_text + "\n"
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
    
    return text.strip()

def extract_text_from_image(file_path):
    """
    Extract text from image with optional preprocessing for better OCR accuracy (optimized)
    """
    try:
        # Try direct OCR first (faster)
        image = Image.open(file_path)
        text = ocr_with_multiple_psm(image)
        
        # If direct OCR result is poor, try with preprocessing
        if not text or len(text) < 20:
            processed_path = preprocess_image_for_ocr(file_path)
            if processed_path and os.path.exists(processed_path):
                text = ocr_with_multiple_psm(Image.open(processed_path))
                # Clean up processed image
                if os.path.exists(processed_path):
                    os.remove(processed_path)

        return text.strip()
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        # Final fallback
        try:
            image = Image.open(file_path)
            text = ocr_with_multiple_psm(image)
            return text.strip() if text else ''
        except Exception as e2:
            print(f"Final OCR fallback failed: {e2}")
            return ''


def advanced_ocr_retry(image_path):
    """Second chance OCR for difficult images (resized + high-contrast)."""
    try:
        img = Image.open(image_path).convert('L')
        # resize for better OCR in low-resolution scans
        w, h = img.size
        img = img.resize((min(3000, int(w * 1.6)), min(3000, int(h * 1.6))), Image.BICUBIC)

        # try a stronger binary threshold
        np_img = np.array(img)
        _, thresh = cv2.threshold(np_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(Image.fromarray(thresh), config='--psm 6 --oem 3 -l eng')
        text = ' '.join(text.split())
        if text:
            return text
    except Exception as e:
        print(f"advanced_ocr_retry failed: {e}")

    return ''


def extract_text_from_file(file_path):
    """
    Main function to extract text from various file formats with OCR optimization
    """
    if not os.path.exists(file_path):
        file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        print(f"extract_text_from_file: file not found {file_path}")
        return ""

    ext = file_path.rsplit('.', 1)[1].lower()
    text = ""
    
    if ext == 'txt':
        # For text files, just read directly
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
    elif ext == 'pdf':
        text = extract_text_from_pdf(file_path)
    elif ext in ['png', 'jpg', 'jpeg']:
        text = extract_text_from_image(file_path)
    
    # Clean up extracted text: remove extra spaces and newlines
    text = ' '.join(text.split())

    # If image had no text, try advanced OCR retry
    if not text and ext in ['png', 'jpg', 'jpeg']:
        text = advanced_ocr_retry(file_path)

    # If text is still empty and pdfplumber is available, try a final PDF text extraction attempt
    if not text and ext == 'pdf' and pdfplumber is not None:
        try:
            with pdfplumber.open(file_path) as pdf:
                pdf_text = ' '.join([p.extract_text() or '' for p in pdf.pages])
                pdf_text = ' '.join(pdf_text.split())
                if pdf_text:
                    text = pdf_text
        except Exception as e:
            print(f"pdfplumber final fallback failed: {e}")

    return text

# New routes for uploading model answer keys
@app.route('/teacher/upload_model_key/<int:test_id>', methods=['GET', 'POST'])
def upload_model_key(test_id):
    if 'teacher_logged_in' in session:
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = f"{datetime.now():%Y%m%d%H%M%S%f}_{secure_filename(file.filename)}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                cur = mysql.connection.cursor()
                cur.execute("INSERT INTO ModelAnswerKeys (test_id, file_path) VALUES (%s, %s)", (test_id, file_path))
                mysql.connection.commit()
                cur.close()
                flash('Model answer key uploaded successfully')
                return redirect(url_for('teacher_home'))
        return render_template('upload_model_key.html', test_id=test_id)
    else:
        return redirect(url_for('teacher_login'))

# Route for students to choose test for uploading answer sheet
@app.route('/student/upload_test')
def upload_test():
    if 'student_logged_in' in session:
        cur = mysql.connection.cursor()
        cur.execute("SELECT t.test_id, t.test_name FROM Tests t JOIN ModelAnswerKeys mak ON t.test_id = mak.test_id")
        tests = cur.fetchall()
        cur.close()
        return render_template('student_upload_test.html', tests=tests)
    else:
        return redirect(url_for('student_login'))
# Route for students to upload answer sheet
@app.route('/student/upload_answer_sheet/<int:test_id>', methods=['GET', 'POST'])
def upload_answer_sheet(test_id):
    if 'student_logged_in' in session:
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = f"{datetime.now():%Y%m%d%H%M%S%f}_{secure_filename(file.filename)}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                extracted_text = extract_text_from_file(file_path)
                # Get model key for the test
                cur = mysql.connection.cursor()
                cur.execute("SELECT file_path FROM ModelAnswerKeys WHERE test_id = %s ORDER BY uploaded_at DESC LIMIT 1", (test_id,))
                model_key = cur.fetchone()
                # Get total marks for the test
                cur.execute("SELECT total_marks FROM Tests WHERE test_id = %s", (test_id,))
                tm = cur.fetchone()
                total_marks = tm[0] if tm else 0
                
                # Get number of questions for marks-per-question calculation
                cur.execute("SELECT COUNT(*) FROM Questions WHERE test_id = %s", (test_id,))
                num_questions = cur.fetchone()[0]
                marks_per_question = total_marks / num_questions if num_questions > 0 else float(total_marks)
                
                score = 0
                evaluation_details = None

                model_text = ""
                if model_key:
                    model_text = extract_text_from_file(model_key[0])

                if not model_text.strip():
                    flash('Warning: Model answer key text is empty or missing. Evaluation may be inaccurate.')

                if not extracted_text.strip():
                    flash('Warning: Student answer sheet text is empty after OCR. Evaluation may be inaccurate.')

                if model_text.strip() and extracted_text.strip() and total_marks > 0:
                    print('DEBUG model_text', repr(model_text)[:500])
                    print('DEBUG student extracted_text', repr(extracted_text)[:500])
                    try:
                        evaluation_result = evaluate_answers(model_text, extracted_text, total_marks)
                        score = round(evaluation_result.get('score', 0), 2)
                        evaluation_details = json.dumps(evaluation_result.get('details', {}))
                    except Exception as e:
                        print(f"Evaluation error: {e}")
                        score = 0
                        evaluation_details = json.dumps({'error': str(e)})
                else:
                    if model_key is None:
                        evaluation_details = json.dumps({'error': 'No model answer key available'})
                    elif total_marks <= 0:
                        evaluation_details = json.dumps({'error': 'Total marks missing or zero'})
                    else:
                        evaluation_details = json.dumps({'error': 'Unable to evaluate due to empty OCR text'})

                cur.execute("INSERT INTO StudentAnswerSheets (student_id, test_id, file_path, extracted_text, score, evaluation_details) VALUES (%s, %s, %s, %s, %s, %s)",
                            (session['student_id'], test_id, file_path, extracted_text, score, evaluation_details))
                mysql.connection.commit()
                cur.close()
                flash('Answer sheet uploaded and evaluated')
                return redirect(url_for('student_home'))
        return render_template('upload_answer_sheet.html', test_id=test_id)
    else:
        return redirect(url_for('student_login'))

# Route for teachers to upload answer sheets for students
@app.route('/teacher/upload_answer_sheet/<int:test_id>', methods=['GET', 'POST'])
def teacher_upload_answer_sheet(test_id):
    if 'teacher_logged_in' in session:
        if request.method == 'POST':
            if 'num_sheets' in request.form and not request.files.getlist('file'):
                num = int(request.form['num_sheets'])
                cur = mysql.connection.cursor()
                cur.execute("SELECT student_id, username FROM Students")
                students = cur.fetchall()
                cur.close()
                return render_template('upload_answer_sheet_teacher.html', test_id=test_id, students=students, num=num)
            else:
                student_ids = request.form.getlist('student_id[]')
                files = request.files.getlist('file[]')
                if not student_ids or not files or len(student_ids) != len(files):
                    flash('Invalid data')
                    return redirect(request.url)
                cur = mysql.connection.cursor()
                # Get total marks for the test
                cur.execute("SELECT total_marks FROM Tests WHERE test_id = %s", (test_id,))
                total_marks_row = cur.fetchone()
                total_marks = total_marks_row[0] if total_marks_row else 0
                
                # Get number of questions for marks-per-question calculation
                cur.execute("SELECT COUNT(*) FROM Questions WHERE test_id = %s", (test_id,))
                num_questions = cur.fetchone()[0]
                marks_per_question = total_marks / num_questions if num_questions > 0 else float(total_marks)

                # Load the latest model answer key once
                cur.execute("SELECT file_path FROM ModelAnswerKeys WHERE test_id = %s ORDER BY uploaded_at DESC LIMIT 1", (test_id,))
                model_key = cur.fetchone()
                model_text = ""
                if model_key:
                    model_key_path = model_key[0]
                    if not os.path.exists(model_key_path):
                        cur.close()
                        flash('Error: The model key file is missing on disk. Please upload a new model key.')
                        return redirect(url_for('upload_model_key', test_id=test_id))
                    model_text = extract_text_from_file(model_key_path)

                if not model_text.strip():
                    cur.close()
                    flash('Error: No model answer key found for this test. Please upload a model key before evaluating sheets.')
                    return redirect(url_for('upload_model_key', test_id=test_id))

                if total_marks <= 0:
                    cur.close()
                    flash('Error: Total marks for this test is missing or invalid.')
                    return redirect(url_for('teacher_home'))

                warnings_collected = set()
                for student_id, file in zip(student_ids, files):
                    if file and allowed_file(file.filename):
                        filename = f"{datetime.now():%Y%m%d%H%M%S%f}_{secure_filename(file.filename)}"
                        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        file.save(file_path)
                        extracted_text = extract_text_from_file(file_path)

                        # If OCR returned empty text, do a stronger retry with higher resolution / fallback
                        file_ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
                        if not extracted_text.strip():
                            if file_ext == 'pdf':
                                extracted_text = extract_text_from_pdf(file_path, dpi=300, max_pages=40)
                            else:
                                extracted_text = advanced_ocr_retry(file_path)

                        score = 0
                        evaluation_details = None

                        if not extracted_text.strip():
                            warnings_collected.add('Some uploaded answer sheets returned empty OCR text. Check image quality and readability.')
                            evaluation_details = json.dumps({'error': 'Unable to evaluate due to empty OCR text'})

                        if not evaluation_details:
                            try:
                                evaluation_result = evaluate_answers(model_text, extracted_text, total_marks)
                                score = round(evaluation_result.get('score', 0), 2)
                                evaluation_details = json.dumps(evaluation_result.get('details', {}))
                            except Exception as e:
                                print(f"Evaluation error: {e}")
                                score = 0
                                evaluation_details = json.dumps({'error': str(e)})

                        cur.execute("INSERT INTO StudentAnswerSheets (student_id, test_id, file_path, extracted_text, score, evaluation_details) VALUES (%s, %s, %s, %s, %s, %s)",
                                    (student_id, test_id, file_path, extracted_text, score, evaluation_details))
                mysql.connection.commit()
                cur.close()
                for warning_message in warnings_collected:
                    flash(warning_message)
                flash('Answer sheets uploaded and evaluated')
                return redirect(url_for('teacher_home'))
        # Get list of students
        cur = mysql.connection.cursor()
        cur.execute("SELECT student_id, username FROM Students")
        students = cur.fetchall()
        cur.close()
        return render_template('upload_answer_sheet_teacher.html', test_id=test_id, students=students)
    else:
        return redirect(url_for('teacher_login'))


###############################################
#####################algorithm#################


###########################################
if __name__ == '__main__':
    app.run(debug=True)