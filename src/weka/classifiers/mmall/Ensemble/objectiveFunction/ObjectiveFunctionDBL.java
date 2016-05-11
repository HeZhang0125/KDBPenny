package weka.classifiers.mmall.Ensemble.objectiveFunction;

import lbfgsb.DifferentiableFunction;
import lbfgsb.FunctionValues;

import weka.classifiers.mmall.Ensemble.DBL;

public abstract class ObjectiveFunctionDBL implements DifferentiableFunction {

	protected final DBL algorithm;
	
	public ObjectiveFunctionDBL(DBL algorithm) {
		this.algorithm = algorithm;
	}

	@Override
	abstract public FunctionValues getValues(double params[]);	
	
	public void finish(){
		
	}
	
}
