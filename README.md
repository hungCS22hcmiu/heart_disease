# Heart Disease Data Mining Framework

This is a complete data mining framework for heart disease prediction using machine learning algorithms. The framework implements data preprocessing, J48 Decision Tree classification, Random Forest classification, and comprehensive model evaluation.

## Project Structure

```
heart_disease/
├── src/main/java/com/example/heart_disease/
│   ├── Main.java                    # Main entry point
│   ├── common/                      # Shared utilities
│   │   └── DataProcessor.java
│   ├── step1/                       # Data preprocessing
│   │   ├── Preprocessing.java
│   │   ├── DataLoader.java
│   │   ├── DataCleaner.java
│   │   └── DataAnalyzer.java
│   ├── step2/                       # J48 Classification
│   │   ├── Step2Classification.java
│   │   └── J48Algorithm.java
│   ├── step3/                       # Random Forest Classification
│   │   ├── Step3Classification.java
│   │   └── RandomForestAlgorithm.java
│   └── step4/                       # Model Evaluation
│       ├── Step4Evaluation.java
│       ├── CrossValidation.java
│       ├── PerformanceMetrics.java
│       └── ModelComparator.java
├── src/main/resources/              # Data files
│   └── heart_disease.csv
├── docs/output/                     # Output reports
│   ├── Step1.txt
│   ├── Step2.txt
│   ├── Step3.txt
│   └── Step4.txt
└── pom.xml
```

## Building the Project

1. Navigate to the project directory:
```bash
cd /Users/blake/IdeaProjects/heart_disease
```

2. Build the project using Maven:
```bash
mvn clean package
```

This will compile the code and create a JAR file in the `target` directory.

## Running the Project

### Option 1: Run all steps using Main.java

Run with the default dataset (src/main/resources/heart_disease.csv):
```bash
mvn exec:java -Dexec.mainClass="com.example.heart_disease.Main"
```

Run with a custom dataset (absolute or relative path):
```bash
mvn exec:java -Dexec.mainClass="com.example.heart_disease.Main" -Dexec.args="<path_to_your_dataset.csv>"
```

Example:
```bash
mvn exec:java -Dexec.mainClass="com.example.heart_disease.Main" -Dexec.args="/path/to/your/dataset.csv"
```

### Option 2: Run individual steps

**Step 1: Data Preprocessing**
```bash
mvn exec:java -Dexec.mainClass="com.example.heart_disease.step1.Preprocessing"
```

**Step 2: J48 Classification**
```bash
mvn exec:java -Dexec.mainClass="com.example.heart_disease.step2.Step2Classification"
```

**Step 3: Random Forest Classification**
```bash
mvn exec:java -Dexec.mainClass="com.example.heart_disease.step3.Step3Classification"
```

**Step 4: Model Evaluation and Comparison**
```bash
mvn exec:java -Dexec.mainClass="com.example.heart_disease.step4.Step4Evaluation"
```

## Output Files

The framework generates the following outputs:

### Cleaned Data (in `src/main/resources/`)
- `heart_disease_cleaned.csv` - Cleaned data in CSV format
- `heart_disease_cleaned.arff` - Cleaned data in ARFF format for Weka

### Reports (in `docs/output/`)
- `Step1.txt` - Data preprocessing and analysis report
- `Step2.txt` - J48 Decision Tree classification results
- `Step3.txt` - Random Forest classification results
- `Step4.txt` - Model evaluation and comparison report

## Dataset

The framework is designed to work with the heart disease dataset from Kaggle:
- https://www.kaggle.com/datasets/oktayrdeki/heart-disease

The dataset should be in CSV format with features related to heart disease diagnosis.

## Authors

## License
