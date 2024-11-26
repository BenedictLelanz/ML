package learning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import main.Concept;
import main.FeatureVector;

public class NeuralNetwork implements Learner {

    public static final double EPSILON = 0.2;

    public static final double THRESHOLD = 0;

    private final Perceptron[] perceptrons;

    public NeuralNetwork(int inputSize) {
        this.perceptrons = new Perceptron[Concept.values().length];
        for (int i = 0; i < this.perceptrons.length; i++)
            this.perceptrons[i] = new Perceptron(inputSize, Concept.values()[i]);
    }

    @Override
    public void learn(List<FeatureVector> trainingSet) {
        for (FeatureVector fv : trainingSet) {
            for (int i = 0; i < this.perceptrons.length; i++) {
                Perceptron perceptron = this.perceptrons[i];
                // Calculate y and t
                int y = perceptron.calculateOutput(fv);
                int t = fv.getConcept().equals(perceptron.getConcept()) ? 1 : 0;
                if (y == t)
                    continue;
                // Update weights
                for (int j = 0; j < fv.getNumFeatures(); j++) {
                    int featureValue = fv.getFeatureValue(j);
                    double deltaW = EPSILON * featureValue * (t - y);
                    perceptron.updateWeight(j, deltaW);
                }
            }
        }
    }

    @Override
    public Concept classify(FeatureVector example) {
        List<Perceptron> perceptrons = new ArrayList<Perceptron>();
        for (Perceptron perceptron : this.perceptrons)
            perceptrons.add(perceptron);
        Collections.shuffle(perceptrons);
        for (Perceptron perceptron : perceptrons) {
            if (perceptron.calculateOutput(example) == 1) {
                return perceptron.getConcept();
            }
        }
        return null;
    }

}

class Perceptron {

    private final double[] weights;

    private final Concept concept;

    Perceptron(int dimension, Concept concept) {
        this.weights = new double[dimension];
        this.concept = concept;
        for (int i = 0; i < dimension; i++)
            this.weights[i] = (Math.random() - 0.5) * 2; // Zufallswert zwischen -1 und 1
    }

    int calculateOutput(FeatureVector fv) {
        double sum = 0;
        for (int i = 0; i < fv.getNumFeatures(); i++)
            sum += fv.getFeatureValue(i) * this.weights[i];
        return sum > NeuralNetwork.THRESHOLD ? 1 : 0;
    }

    Concept getConcept() {
        return this.concept;
    }

    void updateWeight(int index, double deltaW) {
        this.weights[index] += deltaW;
    }

    @Override
    public String toString() {
        return "Neuron" + Arrays.toString(this.weights);
    }

}
