package com.example.heart_disease;

import weka.core.Instances;

public class Preprocessing {

    public static void main(String[] args) {
        try {
            System.out.println("=== STEP 1: DATA PRE-PROCESSING (CLEAN CSV OUTPUT) ===");

            // Initialize components
            DataLoader loader = new DataLoader();
            DataAnalyzer analyzer = new DataAnalyzer();
            DataCleaner cleaner = new DataCleaner();

            // Step 1: Load data
            String csvPath = "src/main/resources/heart_disease.csv";
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

            // Step 6: Save the entire cleaned dataset as a single CSV file.
            loader.saveAsCSV(cleanedData, "src/main/resources/heart_disease_cleaned.csv");

            // *** REMOVED STEPS 7 & 8: Data splitting and saving train/test files ***
            // Commented out or removed completely to meet your request.

            System.out.println("\n STEP 1 COMPLETED. Final output: heart_disease_cleaned.csv");

        } catch (Exception e) {
            System.err.println(" Error in Step 1: " + e.getMessage());
            e.printStackTrace();
        }
    }
}