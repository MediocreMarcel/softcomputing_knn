package de.hft_stuttgart.softcomputing.knn_oo;

import de.hft_stuttgart.softcomputing.knn_oo.NN.KNN;
import de.hft_stuttgart.softcomputing.knn_oo.reader.Einlesen;

import java.io.File;

public class Main {

    /**
     * Main method
     * @param args starting parameters are not used
     */
    public static void main(String[] args) {
        //read training data from file
        double[][] data = Einlesen.einlesenDiabetes(new File("diabetes.csv"), true);

        //init network parameters
        int dimension    = data[0].length-1; //number of dataset parameters => Number of starting layer neurons
        int[] structureNN = {50, 30, 20, 10};//number of neurons (incl. Bias) per hidden layer
        int maxEpoch = 10000; //number of epochs

        //create network
        KNN network   = new KNN(dimension, structureNN);
        //start training
        network.train(data, maxEpoch);

        //read evaluation data from file
        data = Einlesen.einlesenDiabetes(new File("diabetes.csv"), true);
        //start evaluation
        network.evaluate(data);
    }

}
