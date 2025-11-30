package com.example.heart_disease.runner;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import com.example.heart_disease.classification.randomforest.RandomForestClassifier;
import com.example.heart_disease.classification.crossvalidation.CrossValidation;
import com.example.heart_disease.utils.FileUtils;
import com.example.heart_disease.utils.Logger;

import java.io.PrintWriter;

public class RandomForestRunner {

    public static void main(String[] args) {
        PrintWriter writer = null;
        try {
            String dataPath = args.length > 0 ? args[0] : "src/main/resources/heart_disease_cleaned.arff";
            dataPath = FileUtils.resolveDataPath(dataPath);

            writer = FileUtils.createOutputWriter(FileUtils.getOutputPath("Step3.txt"));
            Logger logger = new Logger(writer);

            logger.log("=== STEP 3: CLASSIFICATION (Random Forest with Balancing) ===\n");

            // 1. Load Data
            DataSource source = new DataSource(dataPath);
            Instances data = source.getDataSet();
            logger.log("✓ Dataset loaded");

            // 2. Prepare data: set class attribute and convert to nominal
            RandomForestClassifier rf = new RandomForestClassifier(logger, true);
            Instances preparedData = rf.prepareData(data);
            logger.log("✓ Class attribute converted to nominal");
            logger.log("");

            // 3.  APPLY RESAMPLE FILTER (Crucial Step)
            // Build Random Forest Classifier with Balancing
            rf.buildModel(preparedData);

            // Get balanced data for evaluation
            Instances balancedData = rf.getBalancedData();

            // Evaluation (10-Fold Cross-Validation on Balanced Data)
            CrossValidation cv = new CrossValidation(logger);
            Evaluation eval = cv.performCrossValidation(rf.getClassifier(), balancedData, 10);

            logger.log(eval.toSummaryString());
            logger.log("Confusion Matrix:");
            logger.log(eval.toMatrixString());
            logger.log("");

            cv.printEvaluationResults(eval, balancedData);

            // 4. Save model to binary file
            rf.saveModel("RANDOMFOREST.model");
            logger.log("");

            logger.log("\n=== STEP 3 COMPLETED SUCCESSFULLY ===");

        } catch (Exception e) {
            System.err.println("❌ Error in Step 3: " + e.getMessage());
            e.printStackTrace();
        } finally {
            if (writer != null) {
                writer.close();
                System.out.println("\n✓ Output saved to: docs/output/Step3.txt");
            }
        }
    }
}

