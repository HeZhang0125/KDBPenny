package weka.classifiers.mmall.Online.Bayes;

import java.util.ArrayList;

/**
 * Listener of the error while refining the model with SGD
 * @author F. Petitjean
 *
 */
public interface LearningListener {
    /**
     * Function to be called every time the parameters are updated by the SGD
     * @param nInstancesSoFar the total number of instances seen so far (t)
     * @param CLL -ln(#classValues)-ln(p(trueClass|...));
     * @param se the squared error (||trueProbVector - predicProbVector||<sub>2</sub>)<sup>2</sup>/#classValues
     * @param error 0.0 if correct classification with threshold at .5, else 1.0
     */
    public void updated(long nInstancesSoFar,double CLL,double se, double error);

	public ArrayList<Double> getErrorRates();

	public ArrayList<Double> getRMSEs();
}
