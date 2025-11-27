package com.example.heart_disease.step1;

import weka.core.Instances;
import java.io.FileWriter;
import java.io.PrintWriter;

public class Preprocessing {

    private static PrintWriter logWriter;

    public static void execute(String csvPath, String cleanedCsvPath, String cleanedArffPath, String outputPath) throws Exception {
        // Initialize log file
        logWriter = new PrintWriter(new FileWriter(outputPath));
        
        logAndPrint("=== STEP 1: DATA PRE-PROCESSING (CLEAN CSV & ARFF OUTPUT) ===");

        // Initialize components
        DataLoader loader = new DataLoader();
        DataAnalyzer analyzer = new DataAnalyzer();
        DataCleaner cleaner = new DataCleaner();

        // Step 1: Load data
        Instances originalData = loader.loadCSV(csvPath);

        // Step 2: Analyze original data
        logAndPrint("\n--- ORIGINAL DATA ANALYSIS ---");
        analyzeAndLog(analyzer, originalData);

        // Step 3: Clean data (Includes imputation, encoding, and scaling)
        Instances cleanedData = cleaner.cleanData(originalData);

        // Step 4: Set class attribute (Important for Weka even if just cleaning)
        cleanedData.setClassIndex(cleanedData.numAttributes() - 1);
        logAndPrint("✓ Class attribute set to: " +
                cleanedData.attribute(cleanedData.classIndex()).name());

        // Step 5: Analyze cleaned data
        logAndPrint("\n--- CLEANED DATA ANALYSIS ---");
        analyzeAndLog(analyzer, cleanedData);

        // Step 6: Save the entire cleaned dataset as both CSV and ARFF files
        loader.saveAsCSV(cleanedData, cleanedCsvPath);
        loader.saveAsARFF(cleanedData, cleanedArffPath);

        logAndPrint("\n STEP 1 COMPLETED.");
        logAndPrint("Final outputs:");
        logAndPrint("  - " + cleanedCsvPath + " (For general use)");
        logAndPrint("  - " + cleanedArffPath + " (For WEKA use)");
        
        if (logWriter != null) {
            logWriter.close();
            System.out.println("\n✓ Output saved to: " + outputPath);
        }
    }

    public static void main(String[] args) {
        try {
            String csvPath = "src/main/resources/heart_disease.csv";
            String cleanedCsvPath = "src/main/resources/heart_disease_cleaned.csv";
            String cleanedArffPath = "src/main/resources/heart_disease_cleaned.arff";
            String outputPath = "docs/output/Step1.txt";
            execute(csvPath, cleanedCsvPath, cleanedArffPath, outputPath);
        } catch (Exception e) {
            System.err.println(" Error in Step 1: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void analyzeAndLog(DataAnalyzer analyzer, Instances data) {
        // Capture analysis output
        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
        java.io.PrintStream ps = new java.io.PrintStream(baos);
        java.io.PrintStream old = System.out;
        System.setOut(ps);
        
        analyzer.performAnalysis(data);
        
        System.out.flush();
        System.setOut(old);
        
        String output = baos.toString();
        logAndPrint(output);
    }

    private static void logAndPrint(String message) {
        System.out.print(message);
        if (logWriter != null) {
            logWriter.print(message);
            logWriter.flush();
        }
    }
}