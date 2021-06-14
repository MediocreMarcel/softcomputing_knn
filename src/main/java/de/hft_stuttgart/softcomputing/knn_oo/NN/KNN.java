package de.hft_stuttgart.softcomputing.knn_oo.NN;

import de.hft_stuttgart.softcomputing.knn_oo.NN.neurons.Neuron;
import de.hft_stuttgart.softcomputing.knn_oo.NN.neurons.StartingLayerNeuron;

import java.util.List;

public class KNN {

    List<Neuron> startingNeurons = null;

    public KNN(int inputCount, int[] numberOfNeuronsPerHiddenLayer) {
        startingNeurons = StartingLayerNeuron.initStartingLayer(inputCount, numberOfNeuronsPerHiddenLayer);
    }

    public void train(double[][] trainingData){
       StartingLayerNeuron.initWeights(startingNeurons);
    }
}
