package com.example.heart_disease.step2;

import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class J48Algorithm {
    
    private J48 tree;
    private long buildTime;
    
    /**
     * Build J48 Decision Tree Classifier
     */
    public void buildClassifier(Instances data) throws Exception {
        long startTime = System.currentTimeMillis();
        
        tree = new J48();
        tree.buildClassifier(data);
        
        buildTime = System.currentTimeMillis() - startTime;
    }
    
    /**
     * Get the built J48 classifier
     */
    public J48 getClassifier() {
        return tree;
    }
    
    /**
     * Get build time in milliseconds
     */
    public long getBuildTime() {
        return buildTime;
    }
    
    /**
     * Get the decision tree as string
     */
    public String getTreeString() {
        return tree.toString();
    }
    
    /**
     * Evaluate using training data (resubstitution)
     */
    public Evaluation evaluateOnTrainingData(Instances data) throws Exception {
        Evaluation evalTrain = new Evaluation(data);
        evalTrain.evaluateModel(tree, data);
        return evalTrain;
    }
}
