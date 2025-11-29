# Heart Disease Data Mining Framework

This is a complete data mining framework for heart disease prediction using machine learning algorithms. The framework implements data preprocessing, J48 Decision Tree classification, Random Forest classification, and comprehensive model evaluation.

## Project Structure

```
heart_disease/
├── src/main/java/com/example/heart_disease/
│   ├── classification/              # Classification algorithms
│   │   ├── ClassifierBase.java      # Base class for classifiers
│   │   ├── crossvalidation/
│   │   │   └── CrossValidation.java
│   │   ├── evaluation/
│   │   │   ├── ModelEvaluator.java
│   │   │   └── ResultsParser.java
│   │   ├── j48/
│   │   │   └── J48Classifier.java
│   │   └── randomforest/
│   │       └── RandomForestClassifier.java
│   ├── preprocessing/               # Data preprocessing
│   │   ├── DataAnalyzer.java
│   │   ├── DataCleaner.java
│   │   └── DataLoader.java
│   ├── runner/                      # Main execution classes
│   │   ├── EvaluationRunner.java
│   │   ├── J48Runner.java
│   │   ├── PreprocessingRunner.java
│   │   ├── RandomForestRunner.java
│   │   └── TotalRunner.java
│   └── utils/                       # Utility classes
│       ├── FileUtils.java
│       └── Logger.java
├── src/main/resources/              # Data files
│   ├── heart_disease.csv
│   ├── heart_disease_cleaned.csv
│   └── heart_disease_cleaned.arff
├── docs/                            # Documentation
│   ├── Project Assignment.2025.2.pdf
│   └── output/                      # Output reports
│       ├── Step1.txt
│       ├── Step2.txt
│       ├── Step3.txt
│       └── Step4.txt
├── DECISIONTREE.model               # Saved J48 model
├── RANDOMFOREST.model               # Saved Random Forest model
└── pom.xml                          # Maven configuration
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
mvn exec:java -Dexec.mainClass="com.example.heart_disease.runner.TotalRunner"
```

Run with a custom dataset (absolute or relative path):
```bash
mvn exec:java -Dexec.mainClass="com.example.heart_disease.runner.TotalRunner" -Dexec.args="<path_to_your_dataset.csv>"
```

Example:
```bash
mvn exec:java -Dexec.mainClass="com.example.heart_disease.runner.Main" -Dexec.args="/path/to/your/dataset.csv"
```

### Option 2: Run individual steps

**Step 1: Data Preprocessing**
```bash
mvn exec:java -Dexec.mainClass="com.example.heart_disease.runner.PreprocessingRunner"
```

**Step 2: J48 Classification**
```bash
mvn exec:java -Dexec.mainClass="com.example.heart_disease.runner.J48Runner"
```

**Step 3: Random Forest Classification**
```bash
mvn exec:java -Dexec.mainClass="com.example.heart_disease.runner.RandomForestRunner"
```

**Step 4: Model Evaluation and Comparison**
```bash
mvn exec:java -Dexec.mainClass="com.example.heart_disease.runner.EvaluationRunner"
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
