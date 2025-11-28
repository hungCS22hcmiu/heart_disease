package com.example.heart_disease;

import com.example.heart_disease.step1.Preprocessing;
import com.example.heart_disease.step2.Step2Classification;
import com.example.heart_disease.step3.Step3Classification;
import com.example.heart_disease.step4.Step4Evaluation;

import java.io.File;
import java.nio.file.Files;
//import java.nio.file.Path;
import java.nio.file.Paths;

public class Main {
    
    public static void main(String[] args) {
        try {
            System.out.println("==========================================================");
            System.out.println("   HEART DISEASE DATA MINING FRAMEWORK");
            System.out.println("==========================================================\n");
            
            // Determine input CSV path
            String inputCsvPath;
            if (args.length > 0) {
                inputCsvPath = args[0];
            } else {
                inputCsvPath = "src/main/resources/heart_disease.csv";
            }
            
            // Convert to absolute path if relative
            File csvFile = new File(inputCsvPath);
            if (!csvFile.isAbsolute()) {
                csvFile = csvFile.getAbsoluteFile();
            }
            
            if (!csvFile.exists()) {
                System.err.println("❌ Error: CSV file not found at: " + csvFile.getAbsolutePath());
                System.err.println("\nUsage: java -jar heart_disease.jar [path/to/dataset.csv]");
                System.err.println("Example: java -jar heart_disease.jar /Users/username/data/heart_disease.csv");
                System.exit(1);
            }
            
            System.out.println("Input CSV file: " + csvFile.getAbsolutePath());
            
            // Define output paths (always save to project's resources and docs folders)
            String projectRoot = System.getProperty("user.dir");
            String resourcesDir = projectRoot + "/src/main/resources";
            String docsOutputDir = projectRoot + "/docs/output";
            
            // Create directories if they don't exist
            Files.createDirectories(Paths.get(resourcesDir));
            Files.createDirectories(Paths.get(docsOutputDir));
            
            String cleanedCsvPath = resourcesDir + "/heart_disease_cleaned.csv";
            String cleanedArffPath = resourcesDir + "/heart_disease_cleaned.arff";
            
            String step1Output = docsOutputDir + "/Step1.txt";
            String step2Output = docsOutputDir + "/Step2.txt";
            String step3Output = docsOutputDir + "/Step3.txt";
            String step4Output = docsOutputDir + "/Step4.txt";
            
            System.out.println("Output directory for reports: " + docsOutputDir);
            System.out.println("Output directory for cleaned data: " + resourcesDir);
            System.out.println("\n==========================================================\n");
            
            // STEP 1: Preprocessing
            System.out.println(">>> Starting Step 1: Data Preprocessing");
            Preprocessing.execute(
                csvFile.getAbsolutePath(),
                cleanedCsvPath,
                cleanedArffPath,
                step1Output
            );
            System.out.println(">>> Step 1 completed successfully!\n");
            
            // STEP 2: J48 Classification
            System.out.println(">>> Starting Step 2: J48 Classification");
            Step2Classification.execute(cleanedArffPath, step2Output);
            System.out.println(">>> Step 2 completed successfully!\n");
            
            // STEP 3: Random Forest Classification
            System.out.println(">>> Starting Step 3: Random Forest Classification");
            Step3Classification.execute(cleanedArffPath, step3Output);
            System.out.println(">>> Step 3 completed successfully!\n");
            
            // STEP 4: Model Evaluation and Comparison
            System.out.println(">>> Starting Step 4: Model Evaluation and Comparison");
            Step4Evaluation.execute(cleanedArffPath, step4Output);
            System.out.println(">>> Step 4 completed successfully!\n");
            
            // Final summary
            System.out.println("\n==========================================================");
            System.out.println("   ALL STEPS COMPLETED SUCCESSFULLY!");
            System.out.println("==========================================================");
            System.out.println("\nGenerated files:");
            System.out.println("  Cleaned Data:");
            System.out.println("    - " + cleanedCsvPath);
            System.out.println("    - " + cleanedArffPath);
            System.out.println("\n  Reports:");
            System.out.println("    - " + step1Output);
            System.out.println("    - " + step2Output);
            System.out.println("    - " + step3Output);
            System.out.println("    - " + step4Output);
            System.out.println("\n==========================================================\n");
            
        } catch (Exception e) {
            System.err.println("\n❌ FATAL ERROR: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
