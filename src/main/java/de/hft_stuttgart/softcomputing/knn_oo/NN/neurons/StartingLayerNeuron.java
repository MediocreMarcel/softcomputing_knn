package de.hft_stuttgart.softcomputing.knn_oo.NN.neurons;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class StartingLayerNeuron extends Neuron {
    private static List<Neuron> currentLayer = new LinkedList<>();

    public StartingLayerNeuron() {

    }

    public StartingLayerNeuron(boolean isBias) {
        this.isBias = isBias;
    }

    public static List<Neuron> initStartingLayer(int dimension, int[] structureNN) {
        //creating bias
        Neuron bias = new StartingLayerNeuron(true);
        bias.output = 1.0;
        currentLayer.add(bias);
        //init starting layer objects
        for (int i = 0; i < dimension; i++) {
            currentLayer.add(new StartingLayerNeuron());
        }

        //create hiddenlayer
        Map<Neuron, Double> childrenMap = HiddenLayerNeuron.initHiddenLayer(createLayerMap(currentLayer), structureNN);

        //set Child for each neuron in this starting layer
        for (Neuron neuron : currentLayer) {
            neuron.children = new HashMap<>(childrenMap);//create a unique map for each neuron
        }

        return currentLayer;
    }

    @Override
    protected void calculateForward() {
        //Not needed in Starting layer
        throw new UnsupportedOperationException();
    }

    @Override
    protected void calculateBackward() {
        //Not needed in Starting layer
        throw new UnsupportedOperationException();
    }
}
