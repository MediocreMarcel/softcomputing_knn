package de.hft_stuttgart.softcomputing.knn_oo.NN.neurons;

import java.util.*;

public abstract class Neuron {

    protected Map<Neuron, Double> parents;
    protected Map<Neuron, Double> children;
    protected boolean isBias = false;
    protected double input;
    protected double output;
    protected double delta;

    protected static final double alpha = 0.4;

    protected static Map<Neuron, Double> createLayerMap(List<Neuron> layer) {
        Map<Neuron, Double> layerMap = new HashMap<>();
        for (Neuron neuron : layer) {
            layerMap.put(neuron, 0.0);
        }
        return layerMap;
    }

    public static double getRandomWeight() {
        double weight = Math.random();
        return Math.random() < 0.5 ? -weight : weight;
    }


    protected void setParentWeight(Neuron neuron, double weight) {
        if (this.parents.containsKey(neuron)) {
            this.parents.put(neuron, weight);
        } else {
            System.exit(1);
        }
    }

    protected void setChildWeight(Neuron neuron, double weight) {
        if (this.children.containsKey(neuron)) {
            this.children.put(neuron, weight);
        } else {
            System.exit(1);
        }
    }

    protected void setParentsWeight(Map<Neuron, Double> map) {
        for (Neuron neuron : map.keySet()) {
            if (this.parents.containsKey(neuron)) {
                this.parents.put(neuron, map.get(neuron));
            } else {
                System.exit(1);
            }
        }
    }

    protected void setChildrensWeight(Map<Neuron, Double> map) {
        for (Neuron neuron : map.keySet()) {
            if (this.children.containsKey(neuron)) {
                this.children.put(neuron, map.get(neuron));
            } else {
                System.exit(1);
            }
        }
    }

    public static void initWeights(List<Neuron> neuronLayer) {
        for (Neuron neuron : neuronLayer) {
            if (neuron instanceof EndingLayerNeuron) {
                return;
            }
            neuron.children.keySet().stream().forEach(x -> {
                double randomWeight = neuron.getRandomWeight();
                neuron.children.put(x, randomWeight);
                x.setParentWeight(neuron, randomWeight);
                initWeights(x);
            });
        }
    }

    public static void initWeights(Neuron currentNeuron) {
        if (currentNeuron instanceof EndingLayerNeuron) {
            return;
        }

        Map<Neuron, Double> neuronChildren = currentNeuron.children;
        List<Neuron> childNeurons = new ArrayList<>(neuronChildren.keySet());
        for (Neuron childNeuron : childNeurons) {
            double randomWeight = getRandomWeight();
            neuronChildren.put(childNeuron, randomWeight);
            childNeuron.setParentWeight(currentNeuron, randomWeight);
            initWeights(childNeurons);
        }
    }

    public void startForwardOnChildren() {
        children.forEach((neuron, weights) -> neuron.calculateForward());
        Neuron firstChild = children.keySet().stream().findFirst().get();
        if (!(firstChild instanceof EndingLayerNeuron)) {
            firstChild.startForwardOnChildren();
        }
    }

    public void startBackwardOnParents() {
        Neuron firstParent = parents.keySet().stream().findFirst().get();
        if (!(firstParent instanceof StartingLayerNeuron)) {
            parents.forEach((neuron, weights) -> neuron.calculateBackward());
            firstParent.startBackwardOnParents();
        } else {
            for (Map.Entry<Neuron, Double> entry : parents.entrySet()) {
                updateWeightOnStartingLayer(entry);
            }
        }
    }

    protected void updateWeight(Map.Entry<Neuron, Double> entry){
        entry.setValue(entry.getValue() - alpha * output * entry.getKey().delta);
    }

    protected void updateWeightOnStartingLayer(Map.Entry<Neuron, Double> entry){
        double weight = entry.getValue() - alpha * entry.getKey().output * delta;
        entry.setValue(weight);
        entry.getKey().setChildWeight(this, weight);
    }


    protected double sigmoidFunction(double x) {
        return (1.0 / (1.0 + Math.exp(-x)));
    }

    protected double derivativeSigmoidFunction(double x) {
        return sigmoidFunction(x) * (1 - sigmoidFunction(x));
    }

    protected abstract void calculateForward();

    protected abstract void calculateBackward();

    public void setInput(double input) {
        this.input = input;
    }

    public void setOutput(double output) {
        this.output = output;
    }
}
