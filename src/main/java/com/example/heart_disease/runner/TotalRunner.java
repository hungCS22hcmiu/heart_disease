package com.example.heart_disease.runner;

public class TotalRunner {

    public static void main(String[] args) {
        System.out.println("========================================");
        System.out.println("  HEART DISEASE PREDICTION FRAMEWORK");
        System.out.println("========================================\n");

        try {
            String dataPath = args.length > 0 ? args[0] : null;

            System.out.println(">>> Running Step 1: Data Preprocessing...");
            if (dataPath != null) {
                PreprocessingRunner.main(new String[]{dataPath});
            } else {
                PreprocessingRunner.main(new String[]{});
            }
            System.out.println(">>> Step 1 completed.\n");

            System.out.println(">>> Running Step 2: J48 Decision Tree...");
            J48Runner.main(new String[]{});
            System.out.println(">>> Step 2 completed.\n");

            System.out.println(">>> Running Step 3: Random Forest...");
            RandomForestRunner.main(new String[]{});
            System.out.println(">>> Step 3 completed.\n");

            System.out.println(">>> Running Step 4: Model Evaluation & Comparison...");
            EvaluationRunner.main(new String[]{});
            System.out.println(">>> Step 4 completed.\n");

            System.out.println("========================================");
            System.out.println("  ALL STEPS COMPLETED SUCCESSFULLY!");
            System.out.println("========================================");
            System.out.println("\nOutput files:");
            System.out.println("  - docs/output/Step1.txt");
            System.out.println("  - docs/output/Step2.txt");
            System.out.println("  - docs/output/Step3.txt");
            System.out.println("  - docs/output/Step4.txt");
            System.out.println("  - src/main/resources/heart_disease_cleaned.csv");
            System.out.println("  - src/main/resources/heart_disease_cleaned.arff");
            System.out.println("\nModels:");
            System.out.println("  - DECISIONTREE.model");
            System.out.println("  - RANDOMFOREST.model");

        } catch (Exception e) {
            System.err.println("Error during execution: " + e.getMessage());
            e.printStackTrace();
        }
    }
}

