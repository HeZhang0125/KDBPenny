package weka.classifiers.mmall.Bayes.objectiveFunction;

import weka.classifiers.mmall.Bayes.wdBayes;
import lbfgsb.DifferentiableFunction;
import lbfgsb.FunctionValues;

public abstract class ObjectiveFunctionCLL implements DifferentiableFunction {

	protected final wdBayes algorithm;
	
	public ObjectiveFunctionCLL(wdBayes algorithm) {
		this.algorithm = algorithm;
	}

	@Override
	abstract public FunctionValues getValues(double params[]);	
	
	public void finish(){
		
	}
	
}