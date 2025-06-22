# 🌍 Image Classification System for CubeSat 🛰️

## 🧠 Project Summary  
CubeSats (small satellites) are highly constrained in memory and processing power. Therefore, it is crucial for the satellite to perform onboard pre-processing and decide autonomously which images are worth transmitting to Earth.

**Problem:**  
How can we enable the satellite to autonomously detect and classify images based on quality, content, and scientific value, while also compressing and preparing them efficiently for transmission?

---

## 🎯 Objectives  
- Detect Earth's horizon in satellite images  
- Identify stars and flickering pixels  
- Classify “good images” (sharp, well-exposed, low noise)  
- Perform intelligent image compression  

---

## 📚 Literature Review

### 🔍 Vision Systems in CubeSats  
- Use of **edge detection algorithms** to detect horizon lines  
- Star identification using **star trackers** and database matching

### 🤖 Machine Learning for Image Classification  
- Use of **Convolutional Neural Networks (CNNs)** for quality and noise classification  
- Efficient compression techniques that retain critical visual data

### 🗜️ Image Compression for Space Systems  
- Use of standards like **JPEG2000**, **CCSDS 122/123** tailored for space communication  
- Design considerations:  
  - High computational efficiency  
  - Retention of important features  
  - Compatibility with limited bandwidth  

---

## 🛠️ Development Plan

| Phase | Description                                                   | Deliverables              |
|-------|----------------------------------------------------------------|---------------------------|
| 1     | Research and define quality/horizon/star detection criteria    | Specification document    |
| 2     | Develop separate modules: horizon detection, star detection, image classification | Initial code & test suite |
| 3     | Implement smart compression algorithm                          | Compression module        |
| 4     | Integrate all modules into a unified simulation system         | Functional prototype      |
| 5     | Test using real satellite image datasets                       | Simulation results        |
| 6     | Prepare final documentation and presentation                   | Report & slides           |

---

## 💻 Tools & Environment  
- Programming Language: `Python`  
- Image Processing Libraries: `OpenCV`  
- Datasets: Public satellite image collections  

---
