package de.hft_stuttgart.softcomputing.knn_oo;

import de.hft_stuttgart.softcomputing.knn_oo.NN.KNN;
import de.hft_stuttgart.softcomputing.knn_oo.reader.Einlesen;

import java.io.File;

public class Main {

    public static void main(String[] args) {
        double[][] data = Einlesen.einlesenDiabetes(new File("diabetes.csv"), true);
        int dimension    = data[0].length-1;
        int[] structureNN = {700, 400, 10};//anzahl Knoten (incl. Bias) pro Hiddenschicht
        int maxEpoche = 10;

        KNN network   = new KNN(dimension, structureNN);
        network.train(data, maxEpoche);

        data = Einlesen.einlesenDiabetes(new File("diabetes.csv"), true);
        network.evaluate(data);
    }

}
