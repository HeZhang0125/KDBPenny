package weka.classifiers.mmall.Ensemble.logDistributionComputation;

import weka.classifiers.mmall.DataStructure.DBL.DBLParameters;
import weka.classifiers.mmall.Utils.plTechniques;

import weka.classifiers.mmall.Ensemble.logDistributionComputation.MAP.DBL1LogDistributionComputerMAP;
import weka.classifiers.mmall.Ensemble.logDistributionComputation.MAP.DBL2LogDistributionComputerMAP;
import weka.classifiers.mmall.Ensemble.logDistributionComputation.MAP.DBL3LogDistributionComputerMAP;
import weka.classifiers.mmall.Ensemble.logDistributionComputation.MAP.DBL4LogDistributionComputerMAP;
import weka.classifiers.mmall.Ensemble.logDistributionComputation.MAP.DBL5LogDistributionComputerMAP;

import weka.classifiers.mmall.Ensemble.logDistributionComputation.W.DBL1LogDistributionComputerW;
import weka.classifiers.mmall.Ensemble.logDistributionComputation.W.DBL2LogDistributionComputerW;
import weka.classifiers.mmall.Ensemble.logDistributionComputation.W.DBL3LogDistributionComputerW;
import weka.classifiers.mmall.Ensemble.logDistributionComputation.W.DBL4LogDistributionComputerW;
import weka.classifiers.mmall.Ensemble.logDistributionComputation.W.DBL5LogDistributionComputerW;

import weka.classifiers.mmall.Ensemble.logDistributionComputation.D.DBL1LogDistributionComputerD;
import weka.classifiers.mmall.Ensemble.logDistributionComputation.D.DBL2LogDistributionComputerD;
import weka.classifiers.mmall.Ensemble.logDistributionComputation.D.DBL3LogDistributionComputerD;
import weka.classifiers.mmall.Ensemble.logDistributionComputation.D.DBL4LogDistributionComputerD;
import weka.classifiers.mmall.Ensemble.logDistributionComputation.D.DBL5LogDistributionComputerD;

import weka.core.Instance;

public abstract class LogDistributionComputerDBL {
	
	public abstract void compute(double[] probs, DBLParameters params, Instance inst);
	
	public static LogDistributionComputerDBL getDistributionComputer(int n, int scheme) {
		switch (n) {
		case 1: return getComputerDBL1(scheme);
		case 2: return getComputerDBL2(scheme);
		case 3: return getComputerDBL3(scheme);
		case 4: return getComputerDBL4(scheme);
		case 5: return getComputerDBL5(scheme);
		default:
			System.err.println("A"+n+"JE not implemented, choosing A1JE");
			return getComputerDBL1(scheme); 
		}
	}
	
	public static LogDistributionComputerDBL getComputerDBL1(int scheme) {
		switch (scheme) {
		case plTechniques.MAP: 		return DBL1LogDistributionComputerMAP.getComputer();
		case plTechniques.dCCBN: 	return DBL1LogDistributionComputerD.getComputer();
		case plTechniques.wCCBN: 	return DBL1LogDistributionComputerW.getComputer();
		//case plTechniques.dCCBNf: 	return DBL1LogDistributionComputerDF.getComputer();
		//case plTechniques.wCCBNf: 	return DBL1LogDistributionComputerWF.getComputer();
		default: 
			System.err.println("Scheme not implemented, resorting to MAP");
			return DBL1LogDistributionComputerMAP.getComputer();
		}
	}
	
	public static LogDistributionComputerDBL getComputerDBL2(int scheme) {
		switch (scheme) {
		case plTechniques.MAP: 		return DBL2LogDistributionComputerMAP.getComputer();
		case plTechniques.dCCBN: 	return DBL2LogDistributionComputerD.getComputer();
		case plTechniques.wCCBN: 	return DBL2LogDistributionComputerW.getComputer();
		//case plTechniques.dCCBNf:	return DBL2LogDistributionComputerDF.getComputer();
		//case plTechniques.wCCBNf:	return DBL2LogDistributionComputerWF.getComputer();
		default: 
			System.err.println("Scheme not implemented, resorting to MAP");
			return DBL2LogDistributionComputerMAP.getComputer();
		}
	}
	
	public static LogDistributionComputerDBL getComputerDBL3(int scheme) {
		switch (scheme) {
		case plTechniques.MAP: 		return DBL3LogDistributionComputerMAP.getComputer();
		case plTechniques.dCCBN: 	return DBL3LogDistributionComputerD.getComputer();
		case plTechniques.wCCBN: 	return DBL3LogDistributionComputerW.getComputer();
		//case plTechniques.dCCBNf: 	return DBL3LogDistributionComputerDF.getComputer();
		//case plTechniques.wCCBNf: 	return DBL3LogDistributionComputerWF.getComputer();
		default: 
			System.err.println("Scheme not implemented, resorting to MAP");
			return DBL3LogDistributionComputerMAP.getComputer();
		}
	}
	
	public static LogDistributionComputerDBL getComputerDBL4(int scheme) {
		switch (scheme) {
		case plTechniques.MAP: 		return DBL4LogDistributionComputerMAP.getComputer();
		case plTechniques.dCCBN: 	return DBL4LogDistributionComputerD.getComputer();
		case plTechniques.wCCBN: 	return DBL4LogDistributionComputerW.getComputer();
		//case plTechniques.dCCBNf: 	return DBL4LogDistributionComputerDF.getComputer();
		//case plTechniques.wCCBNf: 	return DBL4LogDistributionComputerWF.getComputer();
		default: 
			System.err.println("Scheme not implemented, resorting to MAP");
			return DBL4LogDistributionComputerMAP.getComputer();
		}
	}
	
	public static LogDistributionComputerDBL getComputerDBL5(int scheme) {
		switch (scheme) {
		case plTechniques.MAP: 		return DBL5LogDistributionComputerMAP.getComputer();
		case plTechniques.dCCBN: 	return DBL5LogDistributionComputerD.getComputer();
		case plTechniques.wCCBN: 	return DBL5LogDistributionComputerW.getComputer();
		//case plTechniques.dCCBNf: 	return DBL5LogDistributionComputerDF.getComputer();
		//case plTechniques.wCCBNf: 	return DBL5LogDistributionComputerWF.getComputer();
		default: 
			System.err.println("Scheme not implemented, resorting to MAP");
			return DBL5LogDistributionComputerMAP.getComputer();
		}
	}
}
