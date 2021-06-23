package de.hft_stuttgart.softcomputing.knn_oo.NN.neurons;

import java.util.*;

public class HiddenLayerNeuron extends Neuron {

    public HiddenLayerNeuron(Map<Neuron, Double> parents) {
        this.parents = new HashMap<>(parents);
    }

    public HiddenLayerNeuron(Map<Neuron, Double> parents, boolean isBias) {
        this.parents = parents;
        this.isBias = isBias;
    }

    public static Map<Neuron, Double> initHiddenLayer(Map<Neuron, Double> parents, int[] structureNN) {

        List<Neuron> currentLayer = new LinkedList<>();

        for (int i = 1; i < structureNN[0]; i++) {
            currentLayer.add(new HiddenLayerNeuron(parents));
        }

        Map<Neuron, Double> currentLayerMap = createLayerMap(currentLayer);
        Map<Neuron, Double> currentLayerMapWithoutBias = new HashMap<>(currentLayerMap);
        addBias(parents, currentLayer, currentLayerMap);

        if (structureNN.length > 1) {
            Map<Neuron, Double> childrenMap = initHiddenLayer(currentLayerMap, Arrays.copyOfRange(structureNN, 1, structureNN.length));

            //set Child for each neuron in this starting layer
            for (Neuron neuron : currentLayer) {
                neuron.children = new HashMap<>(childrenMap);//create a unique map for each neuron
            }
        } else if (structureNN.length == 1) {
            EndingLayerNeuron endingLayerNeuron = new EndingLayerNeuron(currentLayerMap);

            for (Neuron neuron : currentLayer) {
                Map<Neuron, Double> endingLayerMap = new HashMap<>();
                endingLayerMap.put(endingLayerNeuron, 0.0);
                neuron.children = endingLayerMap;
            }
        }


        return currentLayerMapWithoutBias;
    }

    private static void addBias(Map<Neuron, Double> parents, List<Neuron> currentLayer, Map<Neuron, Double> currentLayerMap) {
        Neuron bias = new HiddenLayerNeuron(parents, true);
        bias.output = 1.0;
        currentLayer.add(bias);
        currentLayerMap.put(bias, getRandomWeight());
    }

    protected void calculateForward() {
        input = 0.0;
        parents.forEach((parentNeuron, weight) -> {
            input += parentNeuron.output * weight;
        });
        output = sigmoidFunction(input);
    }

    @Override
    protected void calculateBackward() {
        if (isBias){
            for (Map.Entry<Neuron, Double> entry : children.entrySet()) {
                updateWeight(entry);
            }
            return;
        }

        double sum = 0.0;
        for (Map.Entry<Neuron, Double> entry : children.entrySet()) {
            Neuron childNeuron = entry.getKey();
            Double weight = entry.getValue();

            sum += weight * childNeuron.delta;

            //update weight
            updateWeight(entry);
            childNeuron.setParentWeight(this, entry.getValue());
        }
        delta = derivativeSigmoidFunction(input) * sum;
    }

}
