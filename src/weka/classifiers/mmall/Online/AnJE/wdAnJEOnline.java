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
 * wdAnJESgd Classifier
 * 
 * wdAnJESgd.java     
 * Code written by: Nayyar Zaidi
 * 
 * Options:
 * -------
 * 
 * -D   Discretize numeric attributes
 * -V 	Verbosity
 * -M   Mini-batch (multi-threaded)
 * -W   Initialize weights to AnJE weights
 * 
 * -S	Structure learning (A1JE, A2JE, A3JE, A4JE, A5JE)
 * -P	Parameter learning (MAP, dCCBN, wCCBN, eCCBN)
 * -E   Objective function to optimize (CLL, MSE)
 * -I   Structure to use (Flat, Indexed, IndexedBig, BitMap)  
 * 
 * -R   Regularization
 * -A   Number of Epochs
 * -B   Eta
 * -L   Lambda of regularization
 * 
 */
package weka.classifiers.mmall.Online.AnJE;

import java.util.Arrays;

import weka.classifiers.mmall.DataStructure.AnJE.wdAnJEParameters;
import weka.classifiers.mmall.DataStructure.AnJE.wdAnJEParametersBitmap;
import weka.classifiers.mmall.DataStructure.AnJE.wdAnJEParametersFlat;
import weka.classifiers.mmall.DataStructure.AnJE.wdAnJEParametersIndexed;
import weka.classifiers.mmall.DataStructure.AnJE.wdAnJEParametersIndexedBig;

import weka.classifiers.mmall.Ensemble.logDistributionComputation.LogDistributionComputerAnJE;
import weka.classifiers.mmall.Utils.SUtils;
import weka.classifiers.mmall.Utils.plTechniques;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.UpdateableClassifier;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.supervised.attribute.Discretize;

public class wdAnJEOnline extends AbstractClassifier implements OptionHandler, UpdateableClassifier {

	private static final long serialVersionUID = 4823531716976859217L;

	private Instances m_Instances;

	int nInstances;
	int nAttributes;
	int nc;
	int[] paramsPerAtt;

	private String m_S = "A1JE"; 						// -S (A1JE, A2JE, A3JE, A4JE, A5JE)
	private String m_P = "MAP"; 						// -P (MAP, dCCBN, wCCBN, eCCBN)
	private String m_E = "CLL"; 						// -E (CLL, MSE)
	private String m_I = "Flat"; 						// -I (Flat, Indexed, IndexedBig, BitMap)
	private String m_X = "None"; 						// -X (None, ChiSqTest, GTest, FisherExactTest)

	private boolean m_Discretization = false; 			// -D
	private boolean m_MVerb = false; 					// -V
	private int m_WeightingInitialization = 0; 			// -W	
	private boolean m_MultiThreaded = false; 			// -M

	private ObjectiveFunctionOnlineCLL function_to_optimize;

	private boolean m_Regularization = false; 			// -R
	private int m_Epochs = 1; 							// -A
	private double m_Eta = 0.1; 						// -B
	private double m_Lambda = 0.01; 					// -L 

	private double[] probs;	
	private int numTuples;

	private Discretize m_Disc = null;
	protected wdAnJEParameters dParameters_;
	private LogDistributionComputerAnJE logDComputer;

	private boolean isFeelders = false;

	private int scheme = 1;

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


		/*
		 * ---------------------------------------------------------------------------------------------
		 * Intitialize data structure
		 * ---------------------------------------------------------------------------------------------
		 */

		if (m_P.equalsIgnoreCase("MAP") || m_P.equalsIgnoreCase("MAP2")) {	
			/*
			 * MAP - Maximum Likelihood Estimates of the Parameters characterzing P(x_i|y)
			 * MAP2 - MLE of parameters characterizing P(y|x_i)
			 */
			scheme = plTechniques.MAP;			

		} else if (m_P.equalsIgnoreCase("dCCBN")) {

			scheme = plTechniques.dCCBN;

		} else if (m_P.equalsIgnoreCase("dCCBNf")) {

			scheme = plTechniques.dCCBNf;

		} else if (m_P.equalsIgnoreCase("wCCBN")) {

			scheme = plTechniques.wCCBN;

		} else if (m_P.equalsIgnoreCase("wCCBNf")) {

			scheme = plTechniques.wCCBNf;

		} else if (m_P.equalsIgnoreCase("wCCBN2")) {

			scheme = plTechniques.wCCBN2;

		} else if (m_P.equalsIgnoreCase("eCCBN")) {
			//TODO						
		} else {
			System.out.println("m_P value should be from set {MAP, dCCBN, wCCBN, dCCBNf, wCCBNf, eCCBN, MAP2, wCCBN2}");
		}

		logDComputer = LogDistributionComputerAnJE.getDistributionComputer(numTuples, scheme);

		if (m_I.equalsIgnoreCase("Flat")) {
			dParameters_ = new wdAnJEParametersFlat(nAttributes, nc, nInstances, paramsPerAtt, scheme, numTuples, m_X);				
		} else if (m_I.equalsIgnoreCase("Indexed")) {
			dParameters_ = new wdAnJEParametersIndexed(nAttributes, nc, nInstances, paramsPerAtt, scheme, numTuples, m_X);				
		} else if (m_I.equalsIgnoreCase("IndexedBig")) {
			dParameters_ = new wdAnJEParametersIndexedBig(nAttributes, nc, nInstances, paramsPerAtt, scheme, numTuples, m_X);				
		} else if (m_I.equalsIgnoreCase("BitMap")) {
			dParameters_ = new wdAnJEParametersBitmap(nAttributes, nc, nInstances, paramsPerAtt, scheme, numTuples, m_X);				
		} else {
			System.out.println("m_I value should be from set {Flat, Indexed, IndexedBig, BitMap}");
		}		

		/*
		 * ---------------------------------------------------------------------------------------------
		 * Create Data Structure by leveraging ONE or TWO pass through the data
		 * (These routines are common to all parameter estimation methods)
		 * ---------------------------------------------------------------------------------------------
		 */
		if (nInstances > 0) {
			
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

			/*
			 * Routine specific operations.
			 */

			System.out.println("All data structures are initialized. Starting to estimate parameters.");

			if (m_P.equalsIgnoreCase("MAP")) {			

				/* 
				 * ------------------------------------------------------------------------------
				 * MAP - Maximum Likelihood Estimates of the Parameters characterzing P(x_i|y)
				 * MAP2 - MLE of parameters characterizing P(y|x_i)
				 * ------------------------------------------------------------------------------
				 */

				if (m_P.equalsIgnoreCase("MAP2"))
					dParameters_.convertToProbs_Cond();
				else
					dParameters_.convertToProbs();			

			} else if (m_P.equalsIgnoreCase("dCCBN")) {

				dParameters_.convertToProbs();		

				dParameters_.initializeParameters_D(m_WeightingInitialization, isFeelders);

				function_to_optimize = new ObjectiveFunctionOnlineCLL_d(this);

				if (m_Eta == -1) {
					System.out.println("Value of m_Eta set to -1. Using Golden section search to determine the best value of m_Eta");
					determineEtaValByCV();
					//determineEtaValByCV_GoldenSection();
				}

				int t = 0;		
				double[][] results = new double[m_Epochs][3];

				for (int i = 0; i < m_Epochs; i++) {				
					for (int ii = 0; ii < nInstances; ii++) {
						function_to_optimize.update(m_Instances.instance(ii), t, results[i]);
						t = t + 1;
					}
					if (m_MVerb) {
						System.out.println("Iteration: " + i);
					}
				}

				if (m_MVerb) {
					System.out.print("fxNLL = [");
					for (int i = 0; i < m_Epochs; i++) {
						System.out.print(results[i][0] + ", ");
					}
					System.out.println("];");

					System.out.print("fxError = [");
					for (int i = 0; i < m_Epochs; i++) {
						System.out.print(results[i][1]/nInstances + ", ");
					}
					System.out.println("];");

					System.out.print("fxRMSE = [");
					for (int i = 0; i < m_Epochs; i++) {
						System.out.print(results[i][2]/nInstances + ", ");
					}
					System.out.println("];");
				}

			} else if (m_P.equalsIgnoreCase("wCCBN")) {			

				dParameters_.convertToProbs();			

				dParameters_.initializeParameters_W(m_WeightingInitialization, isFeelders);

				function_to_optimize = new ObjectiveFunctionOnlineCLL_w(this);

				if (m_Eta == -1) {
					System.out.println("Value of m_Eta set to -1. Using line search to determine the best value of m_Eta");
					determineEtaValByCV();
					//determineEtaValByCV_GoldenSection();				
				}

				int t = 0;		
				double[][] results = new double[m_Epochs][3];

				for (int i = 0; i < m_Epochs; i++) {				
					for (int ii = 0; ii < nInstances; ii++) {
						function_to_optimize.update(m_Instances.instance(ii), t, results[i]);
						t = t + 1;
					}
					if (m_MVerb) {
						System.out.println("Iteration: " + i);
					}
				}

				if (m_MVerb) {
					System.out.print("fxNLL = [");
					for (int i = 0; i < m_Epochs; i++) {
						System.out.print(results[i][0] + ", ");
					}
					System.out.println("];");

					System.out.print("fxError = [");
					for (int i = 0; i < m_Epochs; i++) {
						System.out.print(results[i][1]/nInstances + ", ");
					}
					System.out.println("];");

					System.out.print("fxRMSE = [");
					for (int i = 0; i < m_Epochs; i++) {
						System.out.print(results[i][2]/nInstances + ", ");
					}
					System.out.println("];");
				}

			} else if (m_P.equalsIgnoreCase("eCCBN")) {
				//TODO				
			} else {
				System.out.println("m_P value should be from set {MAP, dCCBN, wCCBN, eCCBN}");
			}
			
		}

		System.out.println("Training Finished()");
	}
	
	@Override
	public void updateClassifier(Instance instance) throws Exception {
		dParameters_.updateFirstPass(instance);			
	}
	
	public void updateAfterFirstPass(Instance instance) throws Exception {
		dParameters_.updateAfterFirstPass(instance);
	}
	
	public boolean needSecondPass() {
		boolean flag = false;
		dParameters_.finishedFirstPass();
		if (m_I.equalsIgnoreCase("Flat")) {
			flag = false;				
		} else if (m_I.equalsIgnoreCase("Indexed")) {
			flag = true;
		} else if (m_I.equalsIgnoreCase("IndexedBig")) {
			flag = true;	
		} else if (m_I.equalsIgnoreCase("BitMap")) {
			flag = true;
		} else {			
			System.out.println("m_I value should be from set {Flat, Indexed, IndexedBig, BitMap}");
			flag = false;
		}
		return flag;
	}
	
	public boolean needThirdPass() {
		boolean flag = false;
		if (m_P.equalsIgnoreCase("MAP")) {
			dParameters_.convertToProbs();
			flag = false;				
			
		} else if (m_P.equalsIgnoreCase("dCCBN")) {
			dParameters_.convertToProbs();
			dParameters_.initializeParameters_D(m_WeightingInitialization, isFeelders);
			function_to_optimize = new ObjectiveFunctionOnlineCLL_d(this);
			
			flag = true;
		} else if (m_P.equalsIgnoreCase("wCCBN")) {
			dParameters_.convertToProbs();
			dParameters_.initializeParameters_W(m_WeightingInitialization, isFeelders);
			function_to_optimize = new ObjectiveFunctionOnlineCLL_w(this);
			
			flag = true;	
		} else if (m_P.equalsIgnoreCase("eCCBN")) {
			flag = true;
		} else {			
			System.out.println("m_P value should be from set {MAP, dCCBN, wCCBN, eCCBN}");
			flag = false;
		}
		return flag;
	}
	
	public void update_gradient(Instance instance, int t, double[] results) throws Exception {
		function_to_optimize.update(instance, t, results);
	}

	private void determineEtaValByCV() {

		double[] C = {1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2};
		double[][] results = new double[C.length][3];
		
		for (int i = 0; i < C.length; i++) {
			System.out.println("Setting C = " + C[i]);
			initializeParameters();			
			setEta(C[i]);

			int t = 0;
			for (int ii = 0; ii < nInstances; ii++) {
				function_to_optimize.update(m_Instances.instance(ii), t, results[i]);
				t = t + 1;
			}			
		}

		int val = SUtils.findMaxValueLocationInNDMatrix(results,0);
		initializeParameters();
		System.out.println("----> Choosing C = " + C[val]);
		setEta(C[val]);
	}
	
	private double f(double val) {
		initializeParameters();			
		setEta(val);
		double[] results = new double[3];

		int t = 0;
		for (int ii = 0; ii < nInstances; ii++) {
			function_to_optimize.update(m_Instances.instance(ii), t, results);
			t = t + 1;
		}	
		
		return results[0];
	}
	
	private void determineEtaValByCV_GoldenSection() {
		
		double tol = 1e-5;
		double gr = (Math.sqrt(5)-1)/2;
		gr = 0.5;
		double a = Math.log10(1e-6);
		double b = Math.log10(1e6);
		
		double c = b - gr * (b - a);
		double d = a + gr * (b - a);

		while (Math.abs(c - d) > tol) {		
			double fc = f(Math.pow(10,c));
			double fd = f(Math.pow(10,d));
			
			if (fc < fd) {
				b = d;
				d = c;
				c = b - gr * (b - a);				
			} else {
				a = c;
				c = d;
				d = a + gr * (b - a);
			}				
		}	
		
		double C =  Math.pow(10,(b + a)/2);
		System.out.println("Setting C = " + C);
		setEta(C);
	}

	public void initializeParameters() {
		if (m_P.equalsIgnoreCase("dCCBN"))
			dParameters_.initializeParameters_D(m_WeightingInitialization, isFeelders);
		else if (m_P.equalsIgnoreCase("wCCBN"))
			dParameters_.initializeParameters_W(m_WeightingInitialization, isFeelders);
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

		//m_WeightingInitialization = Utils.getFlag('W', options);
		String SW = Utils.getOption('W', options);
		if (SW.length() != 0) {
			// m_S = Integer.parseInt(SK);
			m_WeightingInitialization = Integer.parseInt(SW);
		}

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

		String ME = Utils.getOption('E', options);
		if (ME.length() != 0) {
			m_E = ME;
		}

		String MI = Utils.getOption('I', options);
		if (MI.length() != 0) {
			m_I = MI;
		}
		
		String MX = Utils.getOption('X', options);
		if (MI.length() != 0) {
			m_X = MX;
		}

		m_Regularization = Utils.getFlag('R', options);

		String strB = Utils.getOption('B', options);
		if (strB.length() != 0) {
			m_Eta = (Double.valueOf(strB));
		}

		String strA = Utils.getOption('A', options);
		if (strA.length() != 0) {
			m_Epochs = (Integer.valueOf(strA));
		}

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
		runClassifier(new wdAnJEOnline(), argv);
	}

	public void printParameters() {
		System.out.println(Arrays.toString(dParameters_.getParameters()));
	}

	public String getMS() {
		return m_S;
	}

	public Instances getN_Instances() {
		return m_Instances;
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

	public boolean getRegularization() {
		return m_Regularization;
	}

	public double setEta(double val) {
		return m_Eta = val;
	}

	public double getEta() {
		return m_Eta;
	}

	public double getLambda() {
		return m_Lambda;
	}

	public int getNInstances() {
		return nInstances;
	}

	public void set_m_I(String string) {
		m_I = string;		
	}

	public void set_m_P(String string) {
		m_P = string;		
	}

	public void setLambda(double a) {		
		m_Lambda = a;
	}

	public void setStepsize(double a) {
		m_Eta = a;		
	}

	public void set_m_S(String string) {
		m_S = string;		
	}

	public void setRegularization(boolean a) {
		m_Regularization = a;		
	}

}
