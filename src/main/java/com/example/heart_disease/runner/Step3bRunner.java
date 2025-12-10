package com.example.heart_disease.runner;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import com.example.heart_disease.classification.j48.J48BalancedClassifier;
import com.example.heart_disease.utils.Logger;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;

public class Step3bRunner {
    
    public static void main(String[] args) {
        PrintWriter fileWriter = null;
        Logger logger = null;
        
        try {
            fileWriter = new PrintWriter(new FileWriter("docs/output/step3b.txt"));
            logger = new Logger(fileWriter);
            
            executeStep3b(logger);
            
        } catch (Exception e) {
            System.err.println("❌ Error in Step 3b: " + e.getMessage());
            e.printStackTrace();
            if (logger != null) {
                logger.log("❌ Error: " + e.getMessage());
            }
        } finally {
            if (fileWriter != null) {
                fileWriter.close();
                System.out.println("\n✓ Output saved to: docs/output/step3b.txt");
            }
        }
    }
    
    public static void executeStep3b(Logger logger) throws Exception {
        logger.log("=== STEP 3B: J48 DECISION TREE WITH BALANCED DATA ===\n");
        
        // 1. Load the ARFF dataset
        String datasetPath = "src/main/resources/heart_disease_cleaned.arff";
        DataSource source = new DataSource(datasetPath);
        Instances data = source.getDataSet();
        
        logger.log("✓ Dataset loaded: " + datasetPath);
        logger.log("✓ Number of instances: " + data.numInstances());
        logger.log("✓ Number of attributes: " + data.numAttributes());
        logger.log("");
        
        // 2. Prepare data (set class and convert to nominal)
        J48BalancedClassifier classifier = new J48BalancedClassifier(logger, true);
        data = classifier.prepareData(data);
        
        // 3. Build J48 with balanced data
        classifier.buildModel(data);
        
        // 4. Display the decision tree model
        logger.log("=== Decision Tree Model ===");
        logger.log(classifier.getClassifier().toString());
        logger.log("");
        
        // 5. Get balanced data for evaluation
        Instances balancedData = classifier.getBalancedData();
        
        // 6. Evaluate using 10-fold cross-validation on balanced data
        logger.log("=== 10-Fold Cross-Validation (on Balanced Data) ===");
        long evalStartTime = System.currentTimeMillis();
        
        Evaluation eval = new Evaluation(balancedData);
        eval.crossValidateModel(classifier.getClassifier(), balancedData, 10, new Random(1));
        
        long evalTime = System.currentTimeMillis() - evalStartTime;
        
        logger.log(eval.toSummaryString());
        logger.log("Confusion Matrix:");
        logger.log(eval.toMatrixString());
        logger.log("");
        
        // 7. Performance metrics
        logger.log("=== Performance Metrics ===");
        logger.log(String.format("Correctly Classified: %.2f%%", eval.pctCorrect()));
        logger.log(String.format("Incorrectly Classified: %.2f%%", eval.pctIncorrect()));
        logger.log(String.format("Kappa Statistic: %.4f", eval.kappa()));
        logger.log(String.format("Mean Absolute Error: %.4f", eval.meanAbsoluteError()));
        logger.log(String.format("Root Mean Squared Error: %.4f", eval.rootMeanSquaredError()));
        logger.log("");
        
        // 8. Class-specific metrics
        logger.log("=== Class-Specific Metrics ===");
        for (int i = 0; i < balancedData.numClasses(); i++) {
            logger.log("\nClass " + i + " (" + balancedData.classAttribute().value(i) + "):");
            logger.log(String.format("  Precision: %.4f", eval.precision(i)));
            logger.log(String.format("  Recall: %.4f", eval.recall(i)));
            logger.log(String.format("  F-Measure: %.4f", eval.fMeasure(i)));
            logger.log(String.format("  ROC Area: %.4f", eval.areaUnderROC(i)));
        }
        
        // 9. Timing information
        logger.log("\n=== Timing Information ===");
        logger.log("Build time: " + classifier.getBuildTime() + " ms");
        logger.log("Evaluation time (10-fold CV): " + evalTime + " ms");
        logger.log("Total time: " + (classifier.getBuildTime() + evalTime) + " ms");
        logger.log("");
        
        logger.log("=== STEP 3B COMPLETED SUCCESSFULLY ===");
    }
}

