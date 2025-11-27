package com.example.heart_disease.common;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class DataProcessor {
    
    /**
     * Load dataset from ARFF file
     */
    public Instances loadARFF(String datasetPath) throws Exception {
        DataSource source = new DataSource(datasetPath);
        Instances data = source.getDataSet();
        return data;
    }
    
    /**
     * Set class attribute to last attribute
     */
    public void setClassIndex(Instances data) {
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
    }
    
    /**
     * Convert numeric class to nominal
     */
    public Instances convertClassToNominal(Instances data) throws Exception {
        NumericToNominal convert = new NumericToNominal();
        String[] options = new String[2];
        options[0] = "-R";
        options[1] = "last"; // Convert last attribute (class)
        convert.setOptions(options);
        convert.setInputFormat(data);
        data = Filter.useFilter(data, convert);
        return data;
    }
    
    /**
     * Prepare dataset: load, set class index, convert to nominal
     */
    public Instances prepareDataset(String datasetPath) throws Exception {
        Instances data = loadARFF(datasetPath);
        setClassIndex(data);
        data = convertClassToNominal(data);
        return data;
    }
}
