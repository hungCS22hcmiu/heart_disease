package com.example.heart_disease.step3;

import com.example.heart_disease.common.DataProcessor;
import java.io.FileWriter;
import java.io.PrintWriter;

import weka.classifiers.Evaluation;
import weka.core.Instances; // CHANGED: Import Resample Filter

public class Step3Classification {
    
    private static PrintWriter logWriter;
    
    public static void execute(String arffPath, String outputPath) throws Exception {
        logWriter = new PrintWriter(new FileWriter(outputPath));
        
        logAndPrint("=== STEP 3: CLASSIFICATION (Balanced Random Forest) ===\n");
        
        // 1. Load Data
        DataProcessor processor = new DataProcessor();
        Instances data = processor.loadARFF(arffPath);
        logAndPrint("✓ Dataset loaded");
        
        // 2. Set Class Index
        processor.setClassIndex(data);
        
        // 3. Convert to Nominal
        data = processor.convertClassToNominal(data);
        logAndPrint("✓ Class attribute converted to nominal");
        
        // 4. APPLY RESAMPLE FILTER (Crucial Step)
       
        logAndPrint("\n=== Balancing Dataset (Resampling) ===");
        RandomForestAlgorithm rfAlgo = new RandomForestAlgorithm();
        Instances balancedData = rfAlgo.balanceDataset(data);
        
        logAndPrint("✓ Data Balanced!");
        logAndPrint("  - Original Instances: " + data.numInstances());
        logAndPrint("  - Balanced Instances: " + balancedData.numInstances());
        logAndPrint("");

        
        logAndPrint("=== Building Random Forest on Balanced Data ===");
        rfAlgo.buildClassifier(balancedData);
        
        logAndPrint("✓ Model built successfully!");
        logAndPrint("✓ Build time: " + rfAlgo.getBuildTime() + " ms");
        logAndPrint("");
        
        
        logAndPrint("=== Evaluation on Training Data ===");
        Evaluation eval = rfAlgo.evaluateOnTrainingData(balancedData);
        
        logAndPrint(eval.toSummaryString());
        logAndPrint("Confusion Matrix:");
        logAndPrint(eval.toMatrixString());
        logAndPrint("");
        
        
        logAndPrint("=== Performance Metrics ===");
        logAndPrint(String.format("Accuracy: %.2f%%", eval.pctCorrect()));
        logAndPrint(String.format("Error Rate: %.2f%%", eval.pctIncorrect()));
        logAndPrint(String.format("Kappa Statistic: %.4f", eval.kappa()));
        logAndPrint(String.format("Mean Absolute Error: %.4f", eval.meanAbsoluteError()));
        logAndPrint(String.format("Root Mean Squared Error: %.4f", eval.rootMeanSquaredError()));
        logAndPrint("");
        logAndPrint("\nClass-specific Metrics:");
        for (int i = 0; i < balancedData.numClasses(); i++) {
            logAndPrint("\nClass: " + balancedData.classAttribute().value(i));
            logAndPrint(String.format("  Precision: %.4f", eval.precision(i)));
            logAndPrint(String.format("  Recall: %.4f", eval.recall(i)));
            logAndPrint(String.format("  F-Measure: %.4f", eval.fMeasure(i)));
            logAndPrint(String.format("  ROC Area: %.4f", eval.areaUnderROC(i)));
        }
        
        logAndPrint("\n=== STEP 3 COMPLETED SUCCESSFULLY ===");
        
        if (logWriter != null) {
            logWriter.close();
            System.out.println("\n✓ Output saved to: " + outputPath);
        }
    }
    
    public static void main(String[] args) {
        try {
            String arffPath = "src/main/resources/heart_disease_cleaned.arff";
            String outputPath = "docs/output/step3.txt";
            execute(arffPath, outputPath);
        } catch (Exception e) {
            System.err.println("❌ Error in Step 3: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void logAndPrint(String message) {
        System.out.println(message);
        if (logWriter != null) {
            logWriter.println(message);
            logWriter.flush(); 
        }
    }
}