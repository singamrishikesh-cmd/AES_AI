-- Setup script for Answer-Evaluation-System MySQL schema
-- Run this in a MySQL client (mysql CLI, Workbench, etc.)

CREATE DATABASE IF NOT EXISTS teacher_part;
USE teacher_part;

-- Admin users
CREATE TABLE IF NOT EXISTS Admins (
  admin_id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(255) NOT NULL UNIQUE,
  password VARCHAR(255) NOT NULL
) ENGINE=InnoDB;

-- Students
CREATE TABLE IF NOT EXISTS Students (
  student_id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(255) NOT NULL UNIQUE,
  password VARCHAR(255) NOT NULL
) ENGINE=InnoDB;

-- Teachers
CREATE TABLE IF NOT EXISTS Teachers (
  teacher_id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(255) NOT NULL UNIQUE,
  password VARCHAR(255) NOT NULL
) ENGINE=InnoDB;

-- Tests (created by teachers)
CREATE TABLE IF NOT EXISTS Tests (
  test_id INT AUTO_INCREMENT PRIMARY KEY,
  test_name VARCHAR(255) NOT NULL,
  teacher_id INT NOT NULL,
  total_marks INT DEFAULT 100,
  FOREIGN KEY (teacher_id) REFERENCES Teachers(teacher_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Questions (each belongs to a test)
CREATE TABLE IF NOT EXISTS Questions (
  question_id INT AUTO_INCREMENT PRIMARY KEY,
  question_text TEXT NOT NULL,
  test_id INT NOT NULL,
  FOREIGN KEY (test_id) REFERENCES Tests(test_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Expected answers (one or more per question)
CREATE TABLE IF NOT EXISTS ExpectedAnswers (
  answer_id INT AUTO_INCREMENT PRIMARY KEY,
  answer_text TEXT NOT NULL,
  question_id INT NOT NULL,
  FOREIGN KEY (question_id) REFERENCES Questions(question_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Student answers (one row per student per question)
CREATE TABLE IF NOT EXISTS StudentAnswers (
  answer_id INT AUTO_INCREMENT PRIMARY KEY,
  student_id INT NOT NULL,
  test_id INT NOT NULL,
  question_id INT NOT NULL,
  answer_text TEXT,
  score DOUBLE,
  evaluation_details TEXT,
  FOREIGN KEY (student_id) REFERENCES Students(student_id) ON DELETE CASCADE,
  FOREIGN KEY (test_id) REFERENCES Tests(test_id) ON DELETE CASCADE,
  FOREIGN KEY (question_id) REFERENCES Questions(question_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Uploaded model answer keys
CREATE TABLE IF NOT EXISTS ModelAnswerKeys (
  key_id INT AUTO_INCREMENT PRIMARY KEY,
  test_id INT NOT NULL,
  file_path VARCHAR(255) NOT NULL,
  uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (test_id) REFERENCES Tests(test_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Uploaded student answer sheets
CREATE TABLE IF NOT EXISTS StudentAnswerSheets (
  sheet_id INT AUTO_INCREMENT PRIMARY KEY,
  student_id INT NOT NULL,
  test_id INT NOT NULL,
  file_path VARCHAR(255) NOT NULL,
  extracted_text TEXT,
  score DOUBLE,
  evaluation_details TEXT,
  uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (student_id) REFERENCES Students(student_id) ON DELETE CASCADE,
  FOREIGN KEY (test_id) REFERENCES Tests(test_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Optional mapping table used in code for deletion
CREATE TABLE IF NOT EXISTS teacherstudentrelationship (
  id INT AUTO_INCREMENT PRIMARY KEY,
  teacher_id INT NOT NULL,
  student_id INT NOT NULL,
  FOREIGN KEY (teacher_id) REFERENCES Teachers(teacher_id) ON DELETE CASCADE,
  FOREIGN KEY (student_id) REFERENCES Students(student_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- Example admin
INSERT INTO Admins (username, password) VALUES ('admin', 'admin123')
  ON DUPLICATE KEY UPDATE password = VALUES(password);

-- Example student (for login testing)
INSERT INTO Students (username, password) VALUES ('student1', 'password1')
  ON DUPLICATE KEY UPDATE password = VALUES(password);

-- Example teacher (for login testing)
INSERT INTO Teachers (username, password) VALUES ('teacher1', 'password1')
  ON DUPLICATE KEY UPDATE password = VALUES(password);
