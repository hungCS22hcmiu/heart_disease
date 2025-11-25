package com.example.heart_disease.step3;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances; // CHANGED: Import Resample Filter
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class Step3Classification {
    
    private static PrintWriter logWriter;
    
    public static void main(String[] args) {
        try {
            logWriter = new PrintWriter(new FileWriter("docs/output/Step3_Balanced.txt"));
            
            logAndPrint("=== STEP 3: CLASSIFICATION (Balanced Random Forest) ===\n");
            
            // 1. Load Data
            String datasetPath = "src/main/resources/heart_disease_cleaned.arff";
            DataSource source = new DataSource(datasetPath);
            Instances data = source.getDataSet();
            logAndPrint("✓ Dataset loaded");
            
            // 2. Set Class Index
            if (data.classIndex() == -1) data.setClassIndex(data.numAttributes() - 1);
            
            // 3. Convert to Nominal
            NumericToNominal convert = new NumericToNominal();
            convert.setOptions(new String[]{"-R", "last"});
            convert.setInputFormat(data);
            data = Filter.useFilter(data, convert);
            logAndPrint("✓ Class attribute converted to nominal");
            
            // 4. APPLY RESAMPLE FILTER (Crucial Step)
           
            logAndPrint("\n=== Balancing Dataset (Resampling) ===");
            Resample resample = new Resample();
            resample.setBiasToUniformClass(1.0); 
            resample.setNoReplacement(false);    
            resample.setSampleSizePercent(100);  
            resample.setInputFormat(data);
            
            
            Instances balancedData = Filter.useFilter(data, resample);
            
            logAndPrint("✓ Data Balanced!");
            logAndPrint("  - Original Instances: " + data.numInstances());
            logAndPrint("  - Balanced Instances: " + balancedData.numInstances());
            logAndPrint("");

            
            logAndPrint("=== Building Random Forest on Balanced Data ===");
            long startTime = System.currentTimeMillis();
            
            RandomForest rf = new RandomForest();
            rf.setNumIterations(100);
            rf.buildClassifier(balancedData); 
            
            long buildTime = System.currentTimeMillis() - startTime;
            logAndPrint("✓ Model built successfully!");
            logAndPrint("✓ Build time: " + buildTime + " ms");
            logAndPrint("");
            
            
            logAndPrint("=== Evaluation (10-Fold Cross-Validation on Balanced Data) ===");
            Evaluation eval = new Evaluation(balancedData);
            eval.crossValidateModel(rf, balancedData, 10, new Random(1));
            
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
            
        } catch (Exception e) {
            String errorMsg = "❌ Error in Step 3: " + e.getMessage();
            logAndPrint(errorMsg);
            e.printStackTrace();
            if (logWriter != null) e.printStackTrace(logWriter);
        } finally {
            if (logWriter != null) {
                logWriter.close();
                System.out.println("\n✓ Output saved to: docs/output/step3.txt");
            }
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