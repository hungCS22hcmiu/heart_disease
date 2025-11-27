package com.example.heart_disease.step3;

import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

public class RandomForestAlgorithm {
    
    private RandomForest rf;
    private long buildTime;
    private Instances balancedData;
    
    /**
     * Apply Resample Filter to balance dataset
     */
    public Instances balanceDataset(Instances data) throws Exception {
        // 4. APPLY RESAMPLE FILTER (Crucial Step)
        Resample resample = new Resample();
        resample.setBiasToUniformClass(1.0); 
        resample.setNoReplacement(false);    
        resample.setSampleSizePercent(100);  
        resample.setInputFormat(data);
        
        balancedData = Filter.useFilter(data, resample);
        
        return balancedData;
    }
    
    /**
     * Build Random Forest classifier on balanced data
     */
    public void buildClassifier(Instances data) throws Exception {
        long startTime = System.currentTimeMillis();
        
        rf = new RandomForest();
        rf.setNumIterations(100);
        rf.buildClassifier(data); 
        
        buildTime = System.currentTimeMillis() - startTime;
    }
    
    /**
     * Get the built Random Forest classifier
     */
    public RandomForest getClassifier() {
        return rf;
    }
    
    /**
     * Get build time in milliseconds
     */
    public long getBuildTime() {
        return buildTime;
    }
    
    /**
     * Get balanced data
     */
    public Instances getBalancedData() {
        return balancedData;
    }
    
    /**
     * Evaluate using training data (resubstitution)
     */
    public Evaluation evaluateOnTrainingData(Instances data) throws Exception {
        Evaluation evalTrain = new Evaluation(data);
        evalTrain.evaluateModel(rf, data);
        return evalTrain;
    }
}
