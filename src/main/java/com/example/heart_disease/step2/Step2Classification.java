package com.example.heart_disease.step2;

import com.example.heart_disease.common.DataProcessor;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import java.io.FileWriter;
import java.io.PrintWriter;
// import java.io.IOException;

public class Step2Classification {
    
    private static PrintWriter logWriter;
    
    public static void execute(String arffPath, String outputPath) throws Exception {
        // Initialize log file
        logWriter = new PrintWriter(new FileWriter(outputPath));
        
        logAndPrint("=== STEP 2: CLASSIFICATION ALGORITHM (J48 Decision Tree) ===\n");
        
        // 1. Load the ARFF dataset
        DataProcessor processor = new DataProcessor();
        Instances data = processor.loadARFF(arffPath);
        
        logAndPrint("✓ Dataset loaded: " + arffPath);
        logAndPrint("✓ Number of instances: " + data.numInstances());
        logAndPrint("✓ Number of attributes: " + data.numAttributes());
        logAndPrint("");
        
        // 2. Set class attribute (last attribute)
        processor.setClassIndex(data);
        
        logAndPrint("Class attribute: " + data.classAttribute().name());
        logAndPrint("Class attribute type: " + data.classAttribute().type());
        
        // 3. Convert numeric class to nominal
        logAndPrint("\n=== Converting Class Attribute to Nominal ===");
        data = processor.convertClassToNominal(data);
        
        logAndPrint("✓ Class attribute converted to nominal");
        logAndPrint("✓ Class values: " + data.classAttribute().toString());
        logAndPrint("");
        
        // 4. Build J48 Decision Tree Classifier
        logAndPrint("=== Building J48 Decision Tree Classifier ===");
        J48Algorithm j48 = new J48Algorithm();
        j48.buildClassifier(data);
        
        logAndPrint("✓ Classifier built successfully!");
        logAndPrint("✓ Build time: " + j48.getBuildTime() + " ms");
        logAndPrint("");
        
        // 5. Display the decision tree
        logAndPrint("=== Decision Tree Model ===");
        logAndPrint(j48.getTreeString());
        logAndPrint("");
        
        // 6. Evaluate using training data (resubstitution)
        logAndPrint("=== Evaluation on Training Data ===");
        Evaluation evalTrain = j48.evaluateOnTrainingData(data);
        
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
        
        // Close the log file
        if (logWriter != null) {
            logWriter.close();
            System.out.println("\n✓ Output saved to: " + outputPath);
        }
    }
    
    public static void main(String[] args) {
        try {
            String arffPath = "src/main/resources/heart_disease_cleaned.arff";
            String outputPath = "docs/output/Step2.txt";
            execute(arffPath, outputPath);
        } catch (Exception e) {
            System.err.println("❌ Error in Step 2: " + e.getMessage());
            e.printStackTrace();
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