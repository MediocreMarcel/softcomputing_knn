package de.hft_stuttgart.softcomputing.knn_oo.NN.neurons;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Neuron {

    protected Map<Neuron, Double> parents;
    protected Map<Neuron, Double> children;
    protected boolean isBias = false;
    protected double input;
    protected double output;
    protected double delta;

    protected static Map<Neuron, Double> createLayerMap(List<Neuron> layer) {
        Map<Neuron, Double> layerMap = new HashMap<>();
        for (Neuron neuron : layer) {
            layerMap.put(neuron, 0.0);
        }
        return layerMap;
    }


}
