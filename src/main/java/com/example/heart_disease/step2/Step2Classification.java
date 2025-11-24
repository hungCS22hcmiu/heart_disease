package com.example.heart_disease.step2;

import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import java.io.FileWriter;
import java.io.PrintWriter;
// import java.io.IOException;

public class Step2Classification {
    
    private static PrintWriter logWriter;
    
    public static void main(String[] args) {
        try {
            // Initialize log file
            logWriter = new PrintWriter(new FileWriter("docs/output/Step2.txt"));
            
            logAndPrint("=== STEP 2: CLASSIFICATION ALGORITHM (J48 Decision Tree) ===\n");
            
            // 1. Load the ARFF dataset
            String datasetPath = "src/main/resources/heart_disease_cleaned.arff";
            DataSource source = new DataSource(datasetPath);
            Instances data = source.getDataSet();
            
            logAndPrint("✓ Dataset loaded: " + datasetPath);
            logAndPrint("✓ Number of instances: " + data.numInstances());
            logAndPrint("✓ Number of attributes: " + data.numAttributes());
            logAndPrint("");
            
            // 2. Set class attribute (last attribute)
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }
            
            logAndPrint("Class attribute: " + data.classAttribute().name());
            logAndPrint("Class attribute type: " + data.classAttribute().type());
            
            // 3. Convert numeric class to nominal
            logAndPrint("\n=== Converting Class Attribute to Nominal ===");
            NumericToNominal convert = new NumericToNominal();
            String[] options = new String[2];
            options[0] = "-R";
            options[1] = "last"; // Convert last attribute (class)
            convert.setOptions(options);
            convert.setInputFormat(data);
            data = Filter.useFilter(data, convert);
            
            logAndPrint("✓ Class attribute converted to nominal");
            logAndPrint("✓ Class values: " + data.classAttribute().toString());
            logAndPrint("");
            
            // 4. Build J48 Decision Tree Classifier
            logAndPrint("=== Building J48 Decision Tree Classifier ===");
            long startTime = System.currentTimeMillis();
            
            J48 tree = new J48();
            tree.buildClassifier(data);
            
            long buildTime = System.currentTimeMillis() - startTime;
            logAndPrint("✓ Classifier built successfully!");
            logAndPrint("✓ Build time: " + buildTime + " ms");
            logAndPrint("");
            
            // 5. Display the decision tree
            logAndPrint("=== Decision Tree Model ===");
            logAndPrint(tree.toString());
            logAndPrint("");
            
            // 6. Evaluate using training data (resubstitution)
            logAndPrint("=== Evaluation on Training Data ===");
            Evaluation evalTrain = new Evaluation(data);
            evalTrain.evaluateModel(tree, data);
            
            logAndPrint(evalTrain.toSummaryString());
            logAndPrint("Confusion Matrix:");
            logAndPrint(evalTrain.toMatrixString());
            logAndPrint("");
            
            // 7. Performance metrics
            logAndPrint("=== Performance Metrics ===");
            logAndPrint(String.format("Accuracy: %.2f%%", evalTrain.pctCorrect()));
            logAndPrint(String.format("Error Rate: %.2f%%", evalTrain.pctIncorrect()));
            logAndPrint(String.format("Kappa Statistic: %.4f", evalTrain.kappa()));
            logAndPrint(String.format("Mean Absolute Error: %.4f", evalTrain.meanAbsoluteError()));
            logAndPrint(String.format("Root Mean Squared Error: %.4f", evalTrain.rootMeanSquaredError()));
            logAndPrint("");
            
            logAndPrint("Class-specific Metrics:");
            for (int i = 0; i < data.numClasses(); i++) {
                logAndPrint("\nClass: " + data.classAttribute().value(i));
                logAndPrint(String.format("  Precision: %.4f", evalTrain.precision(i)));
                logAndPrint(String.format("  Recall: %.4f", evalTrain.recall(i)));
                logAndPrint(String.format("  F-Measure: %.4f", evalTrain.fMeasure(i)));
                logAndPrint(String.format("  ROC Area: %.4f", evalTrain.areaUnderROC(i)));
            }
            
            logAndPrint("\n=== STEP 2 COMPLETED SUCCESSFULLY ===");
            
        } catch (Exception e) {
            String errorMsg = "❌ Error in Step 2: " + e.getMessage();
            logAndPrint(errorMsg);
            e.printStackTrace();
            if (logWriter != null) {
                e.printStackTrace(logWriter);
            }
        } finally {
            // Close the log file
            if (logWriter != null) {
                logWriter.close();
                System.out.println("\n✓ Output saved to: docs/output/Step2.txt");
            }
        }
    }
    
    /**
     * Helper method to print to console and write to log file simultaneously
     */
    private static void logAndPrint(String message) {
        System.out.println(message);
        if (logWriter != null) {
            logWriter.println(message);
            logWriter.flush(); // Ensure immediate write
        }
    }
}