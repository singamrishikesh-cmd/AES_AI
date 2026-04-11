#  AI Answer Sheet Evaluator

## 📖 Overview

The **AI Answer Sheet Evaluator** is an intelligent system designed to automate the evaluation of student answer sheets using Artificial Intelligence and Natural Language Processing (NLP).

Traditional evaluation methods are time-consuming and subjective. This system solves that problem by analyzing student responses, comparing them with predefined answer keys, and generating accurate scores automatically.

---

## 🎯 Objective

* Automate answer sheet evaluation
* Reduce manual effort and time
* Provide fair and unbiased grading
* Improve accuracy using AI techniques
* Support multiple subjects and answer types

---

## ✨ Features

* 📝 Automatic evaluation of subjective answers
* 🔍 Keyword-based and semantic analysis
* 📊 Score generation based on similarity
* 📂 Upload student answers and answer keys
* ⚡ Fast processing and results
* 👨‍💼 Admin, 👨‍🏫 Teacher, 🎓 Student modules
* 🧠 AI-powered intelligent grading
* 📄 Supports text, PDF, and image inputs (OCR)

---

## 🛠️ Technologies Used

* **Python** – Core programming language
* **Flask** – Backend framework
* **MySQL (XAMPP)** – Database
* **NLTK / Sentence Transformers** – NLP processing
* **Scikit-learn** – Similarity calculation
* **OpenCV / Tesseract OCR** – Image text extraction
* **HTML, CSS, JavaScript** – Frontend

---

## ⚙️ How It Works

1. User provides:

   * Answer Key (Correct Answer)
   * Student Answer

2. Preprocessing:

   * Tokenization
   * Stopword removal
   * Stemming / Lemmatization

3. Evaluation:

   * Keyword matching
   * Cosine similarity
   * Semantic similarity

4. Output:

   * Score generated based on similarity
   * Result displayed to user

---

## 📊 Evaluation Techniques

* Keyword Matching
* TF-IDF Vectorization
* Cosine Similarity
* Semantic Analysis

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/sai-dharahas07/AES_AI.git
cd AES_AI
```

---

### 2️⃣ Create virtual environment

```bash
python -m venv venv
.\venv\Scripts\activate
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Setup database

* Open phpMyAdmin (XAMPP)
* Create database: `teacher_part`
* Import: `setup.sql`

---

### 5️⃣ Run the project

```bash
python admin.py
```

---

### 6️⃣ Open in browser

```
http://127.0.0.1:5000
```

---

---

## 💡 Advantages

* Saves time for teachers
* Reduces human bias
* Ensures consistent grading
* Handles large number of answer sheets
* Scalable and efficient

---

## ⚠️ Limitations

* May not fully understand complex human expressions
* Requires well-defined answer keys
* Accuracy depends on NLP models

---

## 🔮 Future Enhancements

* Handwritten answer recognition
* Improved OCR accuracy
* Deep learning models
* Multi-language support
* Cloud deployment

---

## 📂 Project Structure

```
app.py
templates/
static/
requirements.txt
setup.sql
```

---

## 🌐 Use Cases

* Schools and colleges
* Online examination systems
* Practice platforms
* Automated grading systems

---

## 👨‍💻 Author

Developed by **Singam Rishikesh**

---

## ⭐ Conclusion

The AI Answer Sheet Evaluator demonstrates how AI and NLP can modernize education by automating evaluation, improving accuracy, and reducing manual effort.

---
