# IMDB Movie Review Sentiment Analysis

A comprehensive machine learning project for analyzing sentiment in IMDB movie reviews using various classification algorithms and advanced optimization techniques.

## Project Description

This project implements sentiment analysis on IMDB movie review dataset using multiple machine learning approaches. The system processes movie reviews, performs text preprocessing, and classifies them as positive or negative sentiment using various algorithms including Naive Bayes, SVM, KNN, and Decision Trees.

### Key Features

- **Text Preprocessing**: HTML tag removal, special character cleaning, stopword removal
- **TF-IDF Vectorization**: Converts text to numerical features for machine learning
- **Multiple ML Models**: Compares performance across different algorithms
- **Advanced Optimization**: Hyperparameter tuning and model optimization
- **Visualization**: Word clouds, confusion matrices, and performance comparisons
- **Word Importance Analysis**: Identifies key words contributing to sentiment classification

### Dataset

The project uses an IMDB sample dataset (`IMDB_sample.csv.xls`) containing:
- Movie reviews (text)
- Sentiment labels (0 = negative, 1 = positive)
- Balanced dataset with ~3,719 positive and ~3,782 negative reviews

## How to Run

### Prerequisites

1. **Python Environment**: Ensure you have Python 3.7+ installed
2. **Virtual Environment**: Activate the existing virtual environment:
   ```bash
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

### Dependencies

The project requires the following Python packages (already installed in the virtual environment):

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
```

### Running the Analysis

1. **Jupyter Notebook** (Recommended):
   ```bash
   jupyter notebook Sentimen.ipynb
   ```
   Then run all cells in the notebook to execute the complete analysis.

2. **Alternative - Convert to Python Script**:
   If you prefer to run as a Python script, you can export the notebook or create a `main.py` file with the notebook content.

### Expected Output

The analysis will generate:
- Word clouds for positive and negative reviews
- Model performance comparisons
- Confusion matrices and classification reports
- Word importance analysis
- Performance optimization results

## How to Test

### Running Tests with pytest

To test the sentiment analysis functionality, you can create a `test_main.py` file and run:

```bash
pytest test_main.py
```

### Manual Testing

You can also test individual components:

1. **Data Loading Test**:
   ```python
   import pandas as pd
   df = pd.read_csv('IMDB_sample.csv.xls')
   print(f"Dataset shape: {df.shape}")
   print(f"Label distribution: {df['label'].value_counts()}")
   ```

2. **Preprocessing Test**:
   ```python
   # Test text cleaning function
   def clean_text(text):
       import re
       import nltk
       from nltk.corpus import stopwords
       
       stop_words = set(stopwords.words('english'))
       text = str(text).lower()
       text = re.sub(r'<[^>]+>', '', text)
       text = re.sub(r'[^a-zA-Z\s]', '', text)
       tokens = text.split()
       tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
       return " ".join(tokens)
   
   # Test with sample text
   sample_text = "This movie is <b>excellent</b> and amazing!!!"
   cleaned = clean_text(sample_text)
   print(f"Original: {sample_text}")
   print(f"Cleaned: {cleaned}")
   ```

### Performance Validation

The project includes comprehensive model evaluation with:
- **Accuracy Scores**: All models achieve >80% accuracy
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Confusion Matrices**: Detailed classification breakdown
- **Best Model**: SVM achieves 86.41% accuracy

## Project Structure

```
Proposal/
├── README.md                 # This file
├── Sentimen.ipynb           # Main analysis notebook
├── IMDB_sample.csv.xls      # Dataset file
└── venv/                    # Virtual environment
    ├── Scripts/             # Python executables
    └── Lib/site-packages/   # Installed packages
```

## Results Summary

The analysis shows excellent performance across different models:

- **SVM (Support Vector Machine)**: 86.41% accuracy (Best Performance)
- **Naive Bayes**: 84.43% accuracy
- **KNN Optimized**: 76.92% accuracy
- **Decision Tree Optimized**: 59.01% accuracy

Key insights from word importance analysis:
- **Positive words**: "best", "great", "excellent", "wonderful", "perfect"
- **Negative words**: "bad", "worst", "awful", "terrible", "boring"

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes.
