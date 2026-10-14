# 🏛️ SchemeSathi-AI-Government-Scheme-Assistant
SchemeSathi AI is an NLP-powered government scheme assistant built with Python, Streamlit, TF-IDF, and Cosine Similarity. It helps users discover relevant Indian government schemes through smart text search, category filtering, eligibility guidance, document information, and official portal links using an interactive, modern dashboard.

# 🏛️ SchemeSathi AI — Government Scheme Assistant

### An NLP-Powered Government Scheme Discovery Application

SchemeSathi AI is a Python-based application designed to help users discover relevant Indian government welfare schemes through an intelligent text-search system. It uses **TF-IDF Vectorization and Cosine Similarity** to identify schemes related to a user's query and presents relevant information through an interactive Streamlit dashboard.

The application provides scheme descriptions, intended beneficiaries, eligibility guidance, document guidance, and links to official information portals.

## 📌 Project Overview

Finding suitable government schemes can be challenging because information is spread across different official websites. SchemeSathi AI aims to make scheme discovery easier by allowing users to describe their requirements in simple language.

The system compares the user's query with information available in its demonstration dataset and ranks matching schemes according to text similarity.

## ✨ Key Features

- 🔎 **Intelligent Scheme Search:** Search using natural-language queries.
- 🧠 **NLP-Based Retrieval:** Uses TF-IDF and Cosine Similarity to rank relevant schemes.
- 🗂️ **Category Filtering:** Explore schemes by category.
- 🎯 **Configurable Results:** Select the number of results to display.
- 👥 **Beneficiary Information:** View the intended beneficiaries of each scheme.
- ✅ **Eligibility Guidance:** Read general eligibility information.
- 📄 **Document Guidance:** View guidance about application documents.
- 🌐 **Official Portal Links:** Open official websites for further verification.
- 📊 **Interactive Dashboard:** View scheme and category metrics.
- 🎨 **Modern UI:** Dark-purple dashboard with colorful gradients, styled cards, and an interactive sidebar.
- ⚡ **Fast Local Retrieval:** Uses precomputed TF-IDF vectors for efficient searching.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Dataset creation and data manipulation |
| Scikit-learn | Text vectorization and similarity calculation |
| TF-IDF Vectorizer | Converts scheme text into numerical features |
| Cosine Similarity | Measures textual similarity between queries and schemes |
| Streamlit | Interactive web application |
| HTML and CSS | Custom dashboard styling |
| Google Colab | Development and experimentation environment |
| GitHub | Source code management and portfolio presentation |

## 🧠 Model and Methodology

SchemeSathi AI uses a **content-based text retrieval approach**, rather than a trained generative AI model.

### How the System Works

🧠 How the System Works
1. 📂 Dataset Preparation: Government scheme information is organized into structured records.
2. 📝 Text Preparation: Relevant fields, including scheme name, category, beneficiaries, description, keywords, and eligibility, are combined into searchable text.
3. 🔢 Text Vectorization: TF-IDF converts the text into numerical feature vectors.
4. 🔍 Query Processing: The user's search query is transformed using the same fitted vectorizer.
5. 📐 Similarity Calculation: Cosine Similarity measures the similarity between the query vector and scheme vectors.
6. 📊 Ranking: Matching schemes are ranked according to their similarity scores.
7. 🔽 Filtering: Category selection and the requested number of results are applied.
8. 🖥️ Result Presentation: The dashboard displays relevant scheme information and official links.

### 🧠 Core NLP Techniques Used
- 📊 Term Frequency–Inverse Document Frequency (TF-IDF)
- 🔤 Unigram and Bigram Features
- 📐 Cosine Similarity
- 🔎 Content-Based Information Retrieval
- 🏆 Ranked Text Matching

## 🔄 Project Workflow

**User Query → Text Vectorization → Cosine Similarity → Similarity Ranking → Category Filtering → Scheme Results → Official Portal Verification**

## 🗂️ Dataset Information

The current prototype uses a **small, manually defined demonstration dataset containing eight scheme records**. It demonstrates the scheme-search workflow and is not intended to represent a complete or continuously updated government scheme database.

🗂️ Dataset Fields
- 🏛️ Scheme Name
- 🗂️ Category
- 👥 Intended Beneficiary
- 📝 Description
- 🔑 Search Keywords
- ✅ Eligibility Guidance
- 📄 Document Guidance
- 🌐 Official Information URL

### Example Scheme Categories

- 🌾 Agriculture
- 🎓 Education
- 🏥 Healthcare
- 🏠 Housing
- 👩 Women and Family Welfare
- 🏪 Business
- 🚀 Entrepreneurship
- 💰 Savings

## 🔍 Example Queries

Users can try queries such as:

- “I am a student looking for a scholarship.”
- “I am a farmer looking for financial support.”
- “I need information about government health schemes.”
- “I want to start a small business.”
- “I am looking for a government housing scheme.”

The application ranks available demonstration records according to textual similarity.

## 📊 Results and Evaluation

The application displays matching schemes with information such as:

- Scheme name and category
- Description and intended beneficiaries
- Eligibility and document guidance
- Text similarity score
- Official information link

**Important:** The similarity score measures textual relevance. It is not a model accuracy score and does not establish that a person qualifies for a scheme.

Formal evaluation would require a labeled test dataset and suitable information-retrieval metrics, such as:

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)

### ⚠️ Limitations & Important Notes

- 📊 **Limited Dataset:** The current dataset contains only eight demonstration government schemes.
- 🌐 **No Live Data Retrieval:** The application does not automatically retrieve real-time government scheme information.
- 🔍 **Search Limitations:** Search quality depends on the text and vocabulary used in the dataset.
- 🧠 **TF-IDF-Based Matching:** The application uses TF-IDF and cosine similarity, which do not provide full semantic understanding of user queries.
- 🔄 **Information Updates:** Eligibility criteria and required documents may change over time.
- ✅ **No Eligibility Guarantee:** A matching result does not confirm eligibility, approval, or entitlement to government benefits.

**Important:** Users should always verify the latest scheme details, eligibility criteria, and required documents through the relevant official government portal before applying.

## 🚀 Future Scope

- 📊 **Dataset Expansion:** Expand the dataset with verified and reliable government scheme information.
- 📍 **State-Wise & District-Wise Discovery:** Enable users to discover schemes based on their state and district.
- 🌐 **Official API Integration:** Integrate official government APIs and reliable data sources wherever available.
- 🌍 **Multilingual Search:** Support English, Hindi, and Marathi to make scheme discovery accessible to more users.
- 🧠 **Semantic Search:** Implement Sentence-BERT embeddings to improve search relevance and understand the meaning behind user queries.
- 🤖 **Conversational AI Interface:** Integrate a suitable language model to provide interactive, conversational assistance.
- 📝 **Guided Eligibility Questionnaire:** Introduce a step-by-step questionnaire to help users explore schemes based on their circumstances.
- 🔄 **Automated Data Updates:** Implement scheduled data updates and source freshness indicators to keep scheme information current.
- 📈 **Improved Model Evaluation:** Evaluate search performance using a labeled benchmark dataset and relevant retrieval metrics.
- 📱 **Accessibility & Responsive Design:** Improve accessibility and optimize the interface for mobile devices, tablets, and desktops.

These enhancements can help transform SchemeSathi AI into a more comprehensive, user-friendly, multilingual, and reliable government scheme discovery platform.

## 🎯 Learning Outcomes

Through this project, I explored and gained practical experience in the following areas:

- 🧠 **NLP-Based Text Retrieval:** Understanding how Natural Language Processing supports text-based information retrieval.
- 🔢 **TF-IDF Feature Extraction:** Converting textual data into numerical feature representations.
- 📐 **Cosine Similarity:** Measuring similarity between user queries and government scheme descriptions.
- 📝 **Text Preprocessing & Feature Representation:** Preparing textual data for efficient search and matching.
- 🐼 **Dataset Management with Pandas:** Structuring, organizing, and processing government scheme data.
- 🖥️ **Interactive Application Development:** Building an interactive user interface using Streamlit.
- 🎨 **Custom UI Styling:** Enhancing the dashboard using HTML, CSS, and colorful visual elements.
- ⚙️ **Practical Machine Learning:** Applying machine-learning techniques to build a useful, user-focused government scheme discovery application.

This project strengthened my understanding of how NLP, information retrieval, and Streamlit can be combined to develop practical AI-powered applications.


## 🔗 Official Information Portals

- [myScheme — Government Scheme Discovery](https://www.myscheme.gov.in/)
- [National Scholarship Portal](https://scholarships.gov.in/)
- [National Health Authority](https://nha.gov.in/)
- [Pradhan Mantri Ujjwala Yojana](https://www.pmuy.gov.in/)
- [India Post](https://www.indiapost.gov.in/)

## 📌 Project Information

- 🏷️ **Project Name:** SchemeSathi AI — Government Scheme Assistant
- 🤖 **Domain:** Artificial Intelligence, Natural Language Processing (NLP), and Information Retrieval
- 🖥️ **Application Framework:** Streamlit
- 🔍 **Search Techniques:** TF-IDF and Cosine Similarity
- 📚 **Project Type:** Educational Prototype
- 🎯 **Primary Goal:** Simplify the discovery of relevant government scheme information through an interactive, text-based search interface.


## 🌟 Conclusion

SchemeSathi AI demonstrates how traditional NLP and information-retrieval techniques can be used to build a practical government scheme discovery application.

By combining TF-IDF, Cosine Similarity, and Streamlit, the project provides a simple way to search and explore relevant scheme information through an interactive dashboard.

The project can be extended into a more comprehensive assistant by incorporating verified data sources, multilingual capabilities, and semantic retrieval.

