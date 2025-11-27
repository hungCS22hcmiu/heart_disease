package com.example.heart_disease.step4;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import java.util.Random;

public class CrossValidation {
    
    /**
     * Perform 10-fold cross-validation
     */
    public Evaluation performCrossValidation(Classifier classifier, Instances data, int folds, int seed) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifier, data, folds, new Random(seed));
        return eval;
    }
    
    /**
     * Perform 10-fold cross-validation with default seed
     */
    public Evaluation performCrossValidation(Classifier classifier, Instances data) throws Exception {
        return performCrossValidation(classifier, data, 10, 1);
    }
}
