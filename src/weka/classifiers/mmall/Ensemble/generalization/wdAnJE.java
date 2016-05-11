/*
 * MMaLL: An open source system for learning from very large data
 * Copyright (C) 2014 Nayyar A Zaidi, Francois Petitjean and Geoffrey I Webb
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 * Please report any bugs to Nayyar Zaidi <nayyar.zaidi@monash.edu>
 */

/*
 * wdAnJE Classifier
 * 
 * wdAnJE.java     
 * Code written by: Nayyar Zaidi, Francois Petitjean
 * 
 * Options:
 * -------
 * 
 * -D   Discretize numeric attributes
 * -V 	Verbosity
 * -M   Multi-threaded
 * -W   Initialize weights to AnJE weights
 * 
 * -S	Structure learning (A1JE, A2JE, A3JE, A4JE, A5JE)
 * -P	Parameter learning (MAP, dCCBN, wCCBN, eCCBN, MAP2, wCCBN2)
 * -I   Structure to use (Flat, Indexed, IndexedBig, BitMap) 
 * -E   Objective function to optimize (CLL, MSE)
 * 
 */
package weka.classifiers.mmall.Ensemble.generalization;

import lbfgsb.Minimizer;
import lbfgsb.Result;
import lbfgsb.StopConditions;
import weka.classifiers.mmall.Utils.SUtils;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.supervised.attribute.Discretize;

public class wdAnJE extends AbstractClassifier implements OptionHandler {

	private static final long serialVersionUID = 4823531716976859217L;

	private Instances m_Instances;

	int nInstances;
	int nAttributes;
	int nc;
	int[] paramsPerAtt;

	private String m_S = "A1JE"; 						// -S (A1JE, A2JE, A3JE, A4JE, A5JE)
	private String m_P = "MAP";  						// -P (MAP, dCCBN, wCCBN, eCCBN, MAP2, wCCBN2)
	private String m_I = "Flat"; 						// -I (Flat, Indexed, IndexedBig, BitMap)

	private boolean m_Discretization = false; 			// -D
	private boolean m_MVerb = false; 					// -V		
	private boolean m_MultiThreaded = false; 			// -M

	private boolean m_Regularization = false; 			// -R
	private double m_Lambda = 0.01; 					// -L 

	private ObjectiveFunction function_to_optimize;

	private double maxGradientNorm = 0.000000000000000000000000000000001;
	private int m_MaxIterations = 10000;				// -C 

	private double[] probs;	
	private int numTuples;

	private Discretize m_Disc = null;
	protected wdAnJEParameters dParameters_;
	private LogDistributionComputerAnJE logDComputer;

	private boolean m_MThreadVerb = false;				// -T

	@Override
	public void buildClassifier(Instances instances) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		// Discretize instances if required
		if (m_Discretization) {
			m_Disc = new Discretize();
			m_Disc.setInputFormat(instances);
			instances = weka.filters.Filter.useFilter(instances, m_Disc);
		}

		// remove instances with missing class
		m_Instances = new Instances(instances);
		m_Instances.deleteWithMissingClass();
		nInstances = m_Instances.numInstances();
		nAttributes = m_Instances.numAttributes() - 1;		
		nc = m_Instances.numClasses();

		probs = new double[nc];		

		paramsPerAtt = new int[nAttributes];
		for (int u = 0; u < nAttributes; u++) {
			paramsPerAtt[u] = m_Instances.attribute(u).numValues();
		}		

		/*
		 * Initialize structure array based on m_S
		 */
		if (m_S.equalsIgnoreCase("A1JE")) {
			// NB
			numTuples = 1;

		} else if (m_S.equalsIgnoreCase("A2JE")) {
			// A2JE			
			numTuples = 2;

		} else if (m_S.equalsIgnoreCase("A3JE")) {
			// A3JE
			numTuples = 3;	

		} else if (m_S.equalsIgnoreCase("A4JE")) {
			// A4JE
			numTuples = 4;	

		} else if (m_S.equalsIgnoreCase("A5JE")) {
			// A5JE
			numTuples = 5;	

		} else {
			System.out.println("m_S value should be from set {A1JE, A2JE, A3JE, A4JE, A5JE}");
		}		

		/* 
		 * ----------------------------------------------------------------------------------------
		 * Start Parameter Learning Process
		 * ----------------------------------------------------------------------------------------
		 */

		// Initialize LBFGS-Solver

		Minimizer alg = new Minimizer();
		StopConditions sc = alg.getStopConditions();
		sc.setMaxGradientNorm(maxGradientNorm);
		sc.setMaxIterations(m_MaxIterations);
		Result result;

		/*
		 * ---------------------------------------------------------------------------------------------
		 * Intitialize data structure
		 * ---------------------------------------------------------------------------------------------
		 */

		logDComputer = LogDistributionComputerAnJE.getDistributionComputer(numTuples);

		if (m_I.equalsIgnoreCase("Flat")) {
			dParameters_ = new wdAnJEParametersFlat(nAttributes, nc, nInstances, paramsPerAtt, numTuples);				
		} else if (m_I.equalsIgnoreCase("Indexed")) {
			dParameters_ = new wdAnJEParametersIndexed(nAttributes, nc, nInstances, paramsPerAtt, numTuples);
		} else if (m_I.equalsIgnoreCase("IndexedBig")) {
			dParameters_ = new wdAnJEParametersIndexedBig(nAttributes, nc, nInstances, paramsPerAtt, numTuples);
		} else if (m_I.equalsIgnoreCase("BitMap")) {
			dParameters_ = new wdAnJEParametersBitmap(nAttributes, nc, nInstances, paramsPerAtt, numTuples);
		}

		/*
		 * ---------------------------------------------------------------------------------------------
		 * Create Data Structure by leveraging ONE or TWO pass through the data
		 * (These routines are common to all parameter estimation methods)
		 * ---------------------------------------------------------------------------------------------
		 */		
		if (m_MultiThreaded) {

			dParameters_.updateFirstPass_m(m_Instances);
			System.out.println("Finished first pass.");

			dParameters_.finishedFirstPass();
			
			if (dParameters_.needSecondPass() ){
				dParameters_.updateAfterFirstPass_m(m_Instances);				
				System.out.println("Finished second pass.");
			}

		} else {

			for (int i = 0; i < nInstances; i++) {
				Instance instance = m_Instances.instance(i);
				dParameters_.updateFirstPass(instance);				
			}
			System.out.println("Finished first pass.");

			dParameters_.finishedFirstPass();	
			
			if (dParameters_.needSecondPass() ){
				for (int i = 0; i < nInstances; i++) {
					Instance instance = m_Instances.instance(i);
					dParameters_.updateAfterFirstPass(instance);				
				}
				System.out.println("Finished second pass.");
			}
		}

		/*
		 * Routine specific operations.
		 */

		System.out.println("All data structures are initialized. Starting to estimate parameters.");


		dParameters_.convertToProbs();			

		if (m_MultiThreaded) {
			function_to_optimize = new ParallelObjectiveFunctionCLL_w(this);
		} else {
			function_to_optimize = new ObjectiveFunctionCLL_w(this);
		}

		/*
		 * Train the classifier on initialized data structure.
		 */

		if (isM_MVerb()) {
			System.out.println();
			System.out.print("fx = [");
			result = alg.run(function_to_optimize, dParameters_.getParameters());
			System.out.println("];");
			//System.out.println(result);

			System.out.println("NoIter = " + result.iterationsInfo.iterations);
			System.out.println();
		} else {
			result = alg.run(function_to_optimize, dParameters_.getParameters());
			System.out.println("NoIter = " + result.iterationsInfo.iterations);
		}
		function_to_optimize.finish();

		// free up some space
		m_Instances = new Instances(m_Instances, 0);
	}

	@Override
	public double[] distributionForInstance(Instance instance) {
		double[] probs = logDistributionForInstance(instance);
		SUtils.exp(probs);
		return probs;
	}	

	public double[] logDistributionForInstance(Instance inst) {
		double[] probs = new double[nc];
		logDistributionForInstance(probs,inst) ;
		return probs;
	}

	public void logDistributionForInstance(double [] probs,Instance inst) {
		logDComputer.compute(probs, dParameters_, inst);
		SUtils.normalizeInLogDomain(probs);
	}

	// ----------------------------------------------------------------------------------
	// Weka Functions
	// ----------------------------------------------------------------------------------

	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		// class
		result.enable(Capability.NOMINAL_CLASS);
		// instances
		result.setMinimumNumberInstances(0);
		return result;
	}

	@Override
	public void setOptions(String[] options) throws Exception {

		m_Discretization = Utils.getFlag('D', options);
		m_MVerb = Utils.getFlag('V', options);

		m_MultiThreaded = Utils.getFlag('M', options);
		m_MThreadVerb = Utils.getFlag('T', options);

		String SK = Utils.getOption('S', options);
		if (SK.length() != 0) {
			// m_S = Integer.parseInt(SK);
			m_S = SK;
		}

		String MP = Utils.getOption('P', options);
		if (MP.length() != 0) {
			// m_P = Integer.parseInt(MP);
			m_P = MP;
		}

		String MI = Utils.getOption('I', options);
		if (MI.length() != 0) {
			m_I = MI;
		}

		String CK = Utils.getOption('C', options);
		if (CK.length() != 0) {
			m_MaxIterations = Integer.parseInt(CK);			
		}

		m_Regularization = Utils.getFlag('R', options);

		String strL = Utils.getOption('L', options);
		if (strL.length() != 0) {
			m_Lambda = (Double.valueOf(strL));
		}

		Utils.checkForRemainingOptions(options);
	}

	@Override
	public String[] getOptions() {
		String[] options = new String[3];
		int current = 0;
		while (current < options.length) {
			options[current++] = "";
		}
		return options;
	}

	public static void main(String[] argv) {
		runClassifier(new wdAnJE(), argv);
	}

	public boolean getRegularization() {
		return m_Regularization;
	}

	public double getLambda() {
		return m_Lambda;
	}

	public String getMS() {
		return m_S;
	}

	public int getNInstances() {
		return nInstances;
	}

	public int getNc() {
		return nc;
	}

	public int getnAttributes() {
		return nAttributes;
	}

	public wdAnJEParameters getdParameters_() {
		return dParameters_;
	}

	public Instances getM_Instances() {
		return m_Instances;
	}

	public boolean isM_MVerb() {
		return m_MVerb;
	}

	public boolean isM_MThreadVerb() {
		return m_MThreadVerb;
	}

}
