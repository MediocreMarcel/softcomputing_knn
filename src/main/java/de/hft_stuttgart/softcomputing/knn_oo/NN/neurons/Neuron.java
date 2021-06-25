package de.hft_stuttgart.softcomputing.knn_oo.NN.neurons;

import java.util.*;

/**
 * Base class for a Neuron
 */
public abstract class Neuron {

    /**
     * List of parent neurons
     */
    protected Map<Neuron, Double> parents;

    /**
     * List of child neurons
     */
    protected Map<Neuron, Double> children;

    /**
     * indicated if the current neuron is bias
     */
    protected boolean isBias = false;

    /**
     * input of neuron
     */
    protected double input;

    /**
     * output of neuron
     */
    protected double output;

    /**
     * delta of neuron
     */
    protected double delta;

    /**
     * alpha rate for ALL neurons
     */
    protected static final double alpha = 0.4;

    /**
     * converts a list of neurons to a neuron->weight map. Per default all weights are 0.
     * @param layer list of neurons
     * @return a map of neurons and weights
     */
    protected static Map<Neuron, Double> createLayerMap(List<Neuron> layer) {
        Map<Neuron, Double> layerMap = new HashMap<>();
        for (Neuron neuron : layer) {
            layerMap.put(neuron, 0.0);
        }
        return layerMap;
    }

    /**
     * calculates a random weight
     * @return random weight
     */
    public static double getRandomWeight() {
        double weight = Math.random();
        return Math.random() < 0.5 ? -weight : weight;
    }

    /**
     * Methods sets the weight of a neuron->weight entry in the parents list.
     * If the parents list does not contain the passed neuron, a system exit with code 1 will be performed. (TODO This should be changed to an exception in the future)
     * @param neuron neuron of which the weight should be changed
     * @param weight weight of the connection
     */
    protected void setParentWeight(Neuron neuron, double weight) {
        if (this.parents.containsKey(neuron)) {
            this.parents.put(neuron, weight);
        } else {
            System.exit(1);
        }
    }

    /**
     * Methods sets the weight of a neuron->weight entry in the child list.
     * If the parents list does not contain the passed neuron, a system exit with code 1 will be performed. (TODO This should be changed to an exception in the future)
     * @param neuron neuron of which the weight should be changed
     * @param weight weight of the connection
     */
    protected void setChildWeight(Neuron neuron, double weight) {
        if (this.children.containsKey(neuron)) {
            this.children.put(neuron, weight);
        } else {
            System.exit(1);
        }
    }

    //we do not know why we designed it with 2 methods, but it works. Recommendation: DO NOT TOUCH! ;)
    /**
     * Starts the weight initialization between a layer and its children
     * @param neuronLayer the layer of which the weights to the children should be
     */
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

    /**
     * Starts the weight initialization between the child layer and its children. The child layer will be the on of the passed neuron
     * @param currentNeuron neuron of which the children and their children should be initialized
     */
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

    /**
     * starts the forward propagation between this neuron and its children
     */
    public void startForwardOnChildren() {
        children.forEach((neuron, weights) -> neuron.calculateForward());
        Neuron firstChild = children.keySet().stream().findFirst().get();
        if (!(firstChild instanceof EndingLayerNeuron)) {
            firstChild.startForwardOnChildren();
        }
    }

    /**
     * starts the backwards propagation on this neuron and its parents
     */
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

    /**
     * Updates the weight on a map (children or parents) entry
     * @param entry entry of which the new weight should be calculated
     */
    protected void updateWeight(Map.Entry<Neuron, Double> entry){
        entry.setValue(entry.getValue() - alpha * output * entry.getKey().delta);
    }

    /**
     * updated the weight of an entry between the starting layer and the first hidden layer
     * @param entry entry of which the weight should be calculated
     */
    protected void updateWeightOnStartingLayer(Map.Entry<Neuron, Double> entry){
        double weight = entry.getValue() - alpha * entry.getKey().output * delta;
        entry.setValue(weight);
        entry.getKey().setChildWeight(this, weight);
    }

    /**
     * sigmoid function
     * @param x value that should be put into the sigmoid function
     * @return calculated value
     */
    protected double sigmoidFunction(double x) {
        return (1.0 / (1.0 + Math.exp(-x)));
    }

    /**
     * derivative sigmoid function
     * @param x that should be put into the derivative sigmoid function
     * @return calculated value
     */
    protected double derivativeSigmoidFunction(double x) {
        return sigmoidFunction(x) * (1 - sigmoidFunction(x));
    }

    protected abstract void calculateForward();

    protected abstract void calculateBackward();

    /**
     * Setter for the output field
     * @param output value that should be setted
     */
    public void setOutput(double output) {
        this.output = output;
    }
}
