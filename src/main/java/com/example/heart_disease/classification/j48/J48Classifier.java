package com.example.heart_disease.classification.j48;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import com.example.heart_disease.classification.ClassifierBase;
import com.example.heart_disease.utils.Logger;

public class J48Classifier extends ClassifierBase {

    public J48Classifier(Logger logger) {
        super(logger, "J48 Decision Tree");
    }

    @Override
    public Classifier createClassifier() throws Exception {
        J48 tree = new J48();
        return tree;
    }

    public void printTree() {
        if (classifier != null) {
            logger.log("=== Decision Tree Model ===");
            logger.log(classifier.toString());
            logger.log("");
        }
    }
}

