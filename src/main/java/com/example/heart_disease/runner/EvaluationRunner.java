package com.example.heart_disease.runner;

import weka.classifiers.Classifier;
import weka.core.SerializationHelper;
import com.example.heart_disease.classification.evaluation.ModelEvaluator;
import com.example.heart_disease.classification.evaluation.ResultsParser;
import com.example.heart_disease.utils.FileUtils;
import com.example.heart_disease.utils.Logger;

import java.io.PrintWriter;
import java.util.Map;

public class EvaluationRunner {

    public static void main(String[] args) {
        PrintWriter writer = null;
        try {
            writer = FileUtils.createOutputWriter(FileUtils.getOutputPath("Step4.txt"));
            Logger logger = new Logger(writer);

            logger.log("=== STEP 4: MODEL EVALUATION AND COMPARISON ===\n");

            // Check if Step 2 and Step 3 results already exist
            String step2File = FileUtils.getOutputPath("Step2.txt");
            String step3File = FileUtils.getOutputPath("Step3.txt");
            java.io.File step2FileObj = new java.io.File(step2File);
            java.io.File step3FileObj = new java.io.File(step3File);

            // Run Step 2 only if Step2.txt doesn't exist
            if (!step2FileObj.exists()) {
                logger.log(">>> Step2.txt not found. Executing Step 2: J48 Decision Tree...");
                J48Runner.main(new String[]{});
                logger.log("✓ Step 2 completed\n");
            } else {
                logger.log("✓ Step2.txt already exists. Skipping Step 2 execution.\n");
            }

            // Run Step 3 only if Step3.txt doesn't exist
            if (!step3FileObj.exists()) {
                logger.log(">>> Step3.txt not found. Executing Step 3: Random Forest...");
                RandomForestRunner.main(new String[]{});
                logger.log("✓ Step 3 completed\n");
            } else {
                logger.log("✓ Step3.txt already exists. Skipping Step 3 execution.\n");
            }

            // Verify models loaded
            logger.log("=== Verifying Trained Models ===");
            Classifier j48Model = (Classifier) SerializationHelper.read("DECISIONTREE.model");
            logger.log("✓ J48 Decision Tree model loaded from DECISIONTREE.model");

            Classifier rfModel = (Classifier) SerializationHelper.read("RANDOMFOREST.model");
            logger.log("✓ Random Forest model loaded from RANDOMFOREST.model");
            logger.log("");

            // Parse results from Step 2 and Step 3 output files
            logger.log("=== Parsing Evaluation Results ===");

            Map<String, Double> j48Metrics = ResultsParser.parseMetrics(step2File);
            Map<String, Double> rfMetrics = ResultsParser.parseMetrics(step3File);

            Map<String, Map<String, Double>> j48ClassMetrics = ResultsParser.parseClassMetrics(step2File);
            Map<String, Map<String, Double>> rfClassMetrics = ResultsParser.parseClassMetrics(step3File);

            logger.log("✓ J48 metrics parsed from Step2.txt");
            logger.log("✓ Random Forest metrics parsed from Step3.txt");
            logger.log("");

            // Use ModelEvaluator to compare results
            ModelEvaluator evaluator = new ModelEvaluator(logger);

            evaluator.printComparisonHeader();
            evaluator.compareMetricsFromMaps(j48Metrics, rfMetrics);
            evaluator.compareClassSpecificFromMaps(j48ClassMetrics, rfClassMetrics);
            evaluator.printConclusionFromMaps(j48Metrics, rfMetrics);

            logger.log("\n=== NOTES ===");
            logger.log("Detailed evaluation results are available in:");
            logger.log("  - docs/output/Step2.txt (J48 Decision Tree)");
            logger.log("  - docs/output/Step3.txt (Random Forest)");

            logger.log("\n=== STEP 4 COMPLETED SUCCESSFULLY ===");

        } catch (Exception e) {
            System.err.println("❌ Error in Step 4: " + e.getMessage());
            e.printStackTrace();
        } finally {
            if (writer != null) {
                writer.close();
                System.out.println("\n✓ Output saved to: docs/output/Step4.txt");
            }
        }
    }
}

