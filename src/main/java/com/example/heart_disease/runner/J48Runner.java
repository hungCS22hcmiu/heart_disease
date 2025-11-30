package com.example.heart_disease.runner;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import com.example.heart_disease.classification.j48.J48Classifier;
import com.example.heart_disease.classification.crossvalidation.CrossValidation;
import com.example.heart_disease.utils.FileUtils;
import com.example.heart_disease.utils.Logger;

import java.io.PrintWriter;

public class J48Runner {

    public static void main(String[] args) {
        PrintWriter writer = null;
        try {
            String dataPath = args.length > 0 ? args[0] : "src/main/resources/heart_disease_cleaned.arff";
            dataPath = FileUtils.resolveDataPath(dataPath);

            writer = FileUtils.createOutputWriter(FileUtils.getOutputPath("Step2.txt"));

            // Initialize log file
            Logger logger = new Logger(writer);

            logger.log("=== STEP 2: CLASSIFICATION ALGORITHM (J48 Decision Tree) ===\n");

            // 1. Load the ARFF dataset
            DataSource source = new DataSource(dataPath);
            Instances data = source.getDataSet();

            logger.log("✓ Dataset loaded: " + dataPath);
            logger.log("✓ Number of instances: " + data.numInstances());
            logger.log("✓ Number of attributes: " + data.numAttributes());
            logger.log("");

            // 2. Prepare data: set class attribute and convert to nominal
            J48Classifier j48 = new J48Classifier(logger);
            Instances preparedData = j48.prepareData(data);

            logger.log("Class attribute: " + preparedData.classAttribute().name());
            logger.log("✓ Class attribute converted to nominal");
            logger.log("✓ Class values: " + preparedData.classAttribute().toString());
            logger.log("");

            // 3. Build J48 Decision Tree Classifier
            j48.buildModel(preparedData);

            // 4. Display the decision tree
            j48.printTree();

            // 5. Save model to binary file
            String modelPath = "DECISIONTREE.model";
            j48.saveModel(modelPath);

            java.io.File modelFile = new java.io.File(modelPath);
            if (modelFile.exists()) {
                logger.log("✓ Model file size: " + modelFile.length() + " bytes");
            }
            logger.log("");

            // 6. Evaluate using 10-fold cross-validation
            CrossValidation cv = new CrossValidation(logger);
            Evaluation evalCV = cv.performCrossValidation(j48.getClassifier(), preparedData, 10);

            logger.log(evalCV.toSummaryString());
            logger.log("Confusion Matrix:");
            logger.log(evalCV.toMatrixString());
            logger.log("");

            // 7. Performance metrics
            cv.printEvaluationResults(evalCV, preparedData);

            logger.log("\n=== STEP 2 COMPLETED SUCCESSFULLY ===");

        } catch (Exception e) {
            System.err.println("❌ Error in Step 2: " + e.getMessage());
            e.printStackTrace();
        } finally {
            if (writer != null) {
                writer.close();
                System.out.println("\n✓ Output saved to: docs/output/Step2.txt");
            }
        }
    }
}

