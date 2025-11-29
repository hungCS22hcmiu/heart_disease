package com.example.heart_disease.runner;

import weka.core.Instances;
import com.example.heart_disease.preprocessing.DataLoader;
import com.example.heart_disease.preprocessing.DataAnalyzer;
import com.example.heart_disease.preprocessing.DataCleaner;
import com.example.heart_disease.utils.FileUtils;
import com.example.heart_disease.utils.Logger;

import java.io.PrintWriter;

public class PreprocessingRunner {

    public static void main(String[] args) {
        PrintWriter writer = null;
        try {
            String inputPath = args.length > 0 ? args[0] : "src/main/resources/heart_disease.csv";
            inputPath = FileUtils.resolveDataPath(inputPath);

            writer = FileUtils.createOutputWriter(FileUtils.getOutputPath("Step1.txt"));
            Logger logger = new Logger(writer);

            logger.log("=== STEP 1: DATA PRE-PROCESSING (CLEAN CSV & ARFF OUTPUT) ===");

            // Initialize components
            DataLoader loader = new DataLoader();
            DataAnalyzer analyzer = new DataAnalyzer();
            DataCleaner cleaner = new DataCleaner();

            // Step 1: Load data
            String csvPath = inputPath;
            Instances originalData = loader.loadCSV(csvPath);

            // Step 2: Analyze original data
            System.out.println("\n--- ORIGINAL DATA ANALYSIS ---");
            analyzer.performAnalysis(originalData);

            // Step 3: Clean data (Includes imputation, encoding, and scaling)
            Instances cleanedData = cleaner.cleanData(originalData);

            // Step 4: Set class attribute (Important for Weka even if just cleaning)
            cleanedData.setClassIndex(cleanedData.numAttributes() - 1);
            System.out.println("✓ Class attribute set to: " +
                    cleanedData.attribute(cleanedData.classIndex()).name());

            // Step 5: Analyze cleaned data
            System.out.println("\n--- CLEANED DATA ANALYSIS ---");
            analyzer.performAnalysis(cleanedData);

            // Step 6: Save the entire cleaned dataset as both CSV and ARFF files
            loader.saveAsCSV(cleanedData, FileUtils.getResourcePath("heart_disease_cleaned.csv"));
            loader.saveAsARFF(cleanedData, FileUtils.getResourcePath("heart_disease_cleaned.arff"));

            logger.log("\n STEP 1 COMPLETED.");
            logger.log("Final outputs:");
            logger.log("  - heart_disease_cleaned.csv (For general use)");
            logger.log("  - heart_disease_cleaned.arff (For WEKA use)");

        } catch (Exception e) {
            System.err.println(" Error in Step 1: " + e.getMessage());
            e.printStackTrace();
        } finally {
            if (writer != null) {
                writer.close();
                System.out.println("\n✓ Output saved to: docs/output/Step1.txt");
            }
        }
    }
}

