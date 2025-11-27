package com.example.heart_disease.step4;

import com.example.heart_disease.common.DataProcessor;
import com.example.heart_disease.step2.J48Algorithm;
import com.example.heart_disease.step3.RandomForestAlgorithm;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import java.io.FileWriter;
import java.io.PrintWriter;

public class Step4Evaluation {
    
    private static PrintWriter logWriter;
    
    public static void execute(String arffPath, String outputPath) throws Exception {
        logWriter = new PrintWriter(new FileWriter(outputPath));
        
        logAndPrint("=== STEP 4: MODEL EVALUATION AND COMPARISON ===\n");
        
        // 1. Load and prepare data
        DataProcessor processor = new DataProcessor();
        Instances data = processor.prepareDataset(arffPath);
        
        logAndPrint("✓ Dataset loaded and prepared");
        logAndPrint("✓ Number of instances: " + data.numInstances());
        logAndPrint("✓ Number of attributes: " + data.numAttributes());
        logAndPrint("✓ Class attribute: " + data.classAttribute().name());
        logAndPrint("");
        
        // 2. Evaluate J48 with 10-fold cross-validation
        logAndPrint("=== Evaluating J48 Decision Tree ===");
        J48Algorithm j48 = new J48Algorithm();
        j48.buildClassifier(data);
        long j48BuildTime = j48.getBuildTime();
        
        CrossValidation cv = new CrossValidation();
        Evaluation j48Eval = cv.performCrossValidation(j48.getClassifier(), data);
        
        logAndPrint("✓ J48 model built and evaluated");
        logAndPrint("✓ Build time: " + j48BuildTime + " ms");
        logAndPrint("\n--- J48 Cross-Validation Results ---");
        logAndPrint(j48Eval.toSummaryString());
        logAndPrint("Confusion Matrix:");
        logAndPrint(j48Eval.toMatrixString());
        logAndPrint(PerformanceMetrics.generateMetrics(j48Eval, data, "J48"));
        
        // 3. Evaluate Random Forest with 10-fold cross-validation
        logAndPrint("\n=== Evaluating Random Forest (Balanced) ===");
        RandomForestAlgorithm rf = new RandomForestAlgorithm();
        Instances balancedData = rf.balanceDataset(data);
        rf.buildClassifier(balancedData);
        long rfBuildTime = rf.getBuildTime();
        
        logAndPrint("✓ Data balanced: " + data.numInstances() + " -> " + balancedData.numInstances() + " instances");
        
        Evaluation rfEval = cv.performCrossValidation(rf.getClassifier(), balancedData);
        
        logAndPrint("✓ Random Forest model built and evaluated");
        logAndPrint("✓ Build time: " + rfBuildTime + " ms");
        logAndPrint("\n--- Random Forest Cross-Validation Results ---");
        logAndPrint(rfEval.toSummaryString());
        logAndPrint("Confusion Matrix:");
        logAndPrint(rfEval.toMatrixString());
        logAndPrint(PerformanceMetrics.generateMetrics(rfEval, balancedData, "Random Forest"));
        
        // 4. Compare models
        logAndPrint("\n" + ModelComparator.compareModels(
                "J48", j48Eval, j48BuildTime,
                "Random Forest", rfEval, rfBuildTime
        ));
        
        logAndPrint("\n=== STEP 4 COMPLETED SUCCESSFULLY ===");
        
        if (logWriter != null) {
            logWriter.close();
            System.out.println("\n✓ Output saved to: " + outputPath);
        }
    }
    
    public static void main(String[] args) {
        try {
            String arffPath = "src/main/resources/heart_disease_cleaned.arff";
            String outputPath = "docs/output/Step4.txt";
            execute(arffPath, outputPath);
        } catch (Exception e) {
            System.err.println("❌ Error in Step 4: " + e.getMessage());
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
