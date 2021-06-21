package de.hft_stuttgart.softcomputing.knn_oo.NN;

import de.hft_stuttgart.softcomputing.knn_oo.NN.neurons.Neuron;
import de.hft_stuttgart.softcomputing.knn_oo.NN.neurons.StartingLayerNeuron;

import java.util.List;

public class KNN {

    public static double[] currentDataRow;
    List<Neuron> startingNeurons = null;

    public KNN(int inputCount, int[] numberOfNeuronsPerHiddenLayer) {
        startingNeurons = StartingLayerNeuron.initStartingLayer(inputCount, numberOfNeuronsPerHiddenLayer);
    }

    public void train(double[][] trainingData) {
        StartingLayerNeuron.initWeights(startingNeurons);
        for (int i = 0; i < trainingData.length; i++) {
            currentDataRow = trainingData[i];
            for (int j = 0; j < trainingData[i].length - 1 /*ignore outcome*/; j++)
                startingNeurons.get(j).setOutput(currentDataRow[j]);
        }
        startingNeurons.get(0).startForwardOnChildren();
    }
}
