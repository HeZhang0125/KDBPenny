package weka.classifiers.mmall.Ensemble.generalization;

import weka.core.Instance;

public abstract class LogDistributionComputerAnJE {
	
	public abstract void compute(double[] probs, wdAnJEParameters params, Instance inst);
	
	public static LogDistributionComputerAnJE getDistributionComputer(int n) {
		switch (n) {
		case 1: return getComputerA1JE();
		case 2: return getComputerA2JE();
		case 3: return getComputerA3JE();
		case 4: return getComputerA4JE();
		case 5: return getComputerA5JE();
		default:
			System.err.println("A"+n+"JE not implemented, choosing A1JE");
			return getComputerA1JE(); 
		}
	}
	
	public static LogDistributionComputerAnJE getComputerA1JE() {
		return A1JELogDistributionComputerW.getComputer();		
	}
	
	public static LogDistributionComputerAnJE getComputerA2JE() {
		return A2JELogDistributionComputerW.getComputer();		
	}
	
	public static LogDistributionComputerAnJE getComputerA3JE() {
		return A3JELogDistributionComputerW.getComputer();		
	}
	
	public static LogDistributionComputerAnJE getComputerA4JE() {
		return A4JELogDistributionComputerW.getComputer();		
	}
	
	public static LogDistributionComputerAnJE getComputerA5JE() {
		return A5JELogDistributionComputerW.getComputer();
		
	}
	
}
