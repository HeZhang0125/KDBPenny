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
 * wdBayes Classifier
 * 
 * wdBayes.java     
 * Code written by: Nayyar Zaidi, Francois Petitjean
 * 
 * Options:
 * -------
 * 
 * -t /Users/nayyar/WData/datasets_DM/shuttle.arff
 * -V -M 
 * -S "chordalysis"
 * -K 2
 * -P "wCCBN"
 * -W 1
 * 
 */
package weka.classifiers.mmall.Bayes;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.mmall.DataStructure.xxyDist;
import weka.classifiers.mmall.DataStructure.Bayes.wdBayesNode;
import weka.classifiers.mmall.DataStructure.Bayes.wdBayesParametersTree;
import weka.classifiers.mmall.Utils.CorrelationMeasures;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.supervised.attribute.Discretize;
import java.io.FileReader;

public class kDBPenny extends AbstractClassifier implements OptionHandler {

	private static final long serialVersionUID = 691858787988950382L;

	public static int nInstances;
	public static Instances m_Instances;
	int nAttributes;
	int nc;
	public int[] paramsPerAtt; // num of att values of each attribute
	xxyDist xxyDist_;

	private Discretize m_Disc = null;

	public wdBayesParametersTree dParameters_;

	private int[][] m_Parents;
	private int[] m_Order;

	private int m_KDB = 1; // -K
	private int N0 = 5;

	private boolean m_MVerb = false; // -V
	private boolean m_Discretization = false; // -D
	public static double laplacePseudocount = 1;

	@Override
	public void buildClassifier(Instances m_Instances) throws Exception {
		
		m_Instances = m_Instances;

		// can classifier handle the data?
		getCapabilities().testWithFail(m_Instances);

		// Discretize instances if required
		if (m_Discretization) {
			m_Disc = new Discretize();
			m_Disc.setInputFormat(m_Instances);
			m_Instances = weka.filters.Filter.useFilter(m_Instances, m_Disc);
		}

		// remove instances with missing class
		m_Instances.deleteWithMissingClass();

		// num of attributes, classes, instances
		nAttributes = m_Instances.numAttributes() - 1;
		nc = m_Instances.numClasses();
		nInstances = m_Instances.numInstances();

		// paramsPerAtt[i]: the num of ith attribute values
		paramsPerAtt = new int[nAttributes];
		for (int u = 0; u < nAttributes; u++) {
			paramsPerAtt[u] = m_Instances.attribute(u).numValues();
		}

		m_Parents = new int[nAttributes][];
		m_Order = new int[nAttributes];
		for (int i = 0; i < nAttributes; i++) {
			getM_Order()[i] = i; // initialize m_Order[i]=i
		}

		double[] mi = null; // mutual information
		double[][] cmi = null; // conditional mutual information

		/*
		 * Initialize kdb structure
		 */

		/**
		 * compute mutual information and conditional mutual information
		 * respectively
		 */

		xxyDist_ = new xxyDist(m_Instances);
		xxyDist_.addToCount(m_Instances);

		mi = new double[nAttributes];
		cmi = new double[nAttributes][nAttributes];
		CorrelationMeasures.getMutualInformation(xxyDist_.xyDist_, mi);// I(Xi;C)
		CorrelationMeasures.getCondMutualInf(xxyDist_, cmi);// I(Xi;Xj|C),i not
															// equal to J
		// Sort attributes on MI with the class
		m_Order = SUtils.sort(mi);

		// Calculate parents based on MI and CMI
		for (int u = 0; u < nAttributes; u++) {
			int nK = Math.min(u, m_KDB);

			if (nK > 0) {
				m_Parents[u] = new int[nK];

				double[] cmi_values = new double[u];
				for (int j = 0; j < u; j++) {
					cmi_values[j] = cmi[m_Order[u]][m_Order[j]];
				}

				int[] cmiOrder = SUtils.sort(cmi_values);

				for (int j = 0; j < nK; j++) {
					m_Parents[u][j] = m_Order[cmiOrder[j]];// cmiOrder[0] is the
															// maximum value
				}
			}
		}

		// Update m_Parents based on m_Order
		int[][] m_ParentsTemp = new int[nAttributes][];
		for (int u = 0; u < nAttributes; u++) {
			if (m_Parents[u] != null) {
				m_ParentsTemp[m_Order[u]] = new int[m_Parents[u].length];
				for (int j = 0; j < m_Parents[u].length; j++) {
					m_ParentsTemp[m_Order[u]][j] = m_Parents[u][j];
				}
			}
		}
		for (int i = 0; i < nAttributes; i++) {
			getM_Order()[i] = i;
		}

		m_Parents = null;
		m_Parents = m_ParentsTemp;
		m_ParentsTemp = null;

		// Print the structure
		// System.out.println("\nm_Order: "+Arrays.toString(m_Order));
		for (int i = 0; i < nAttributes; i++) {
			System.out.print(i + " : ");
			if (m_Parents[i] != null) {
				for (int j = 0; j < m_Parents[i].length; j++) {
					System.out.print(m_Parents[i][j] + ",");
				}
			}
			System.out.println();
		}

		dParameters_ = new wdBayesParametersTree(m_Instances, paramsPerAtt, m_Order, m_Parents, 1);// m_P=1:
																										// MAP

		// Update dTree_ based on parents
		for (int i = 0; i < nInstances; i++) {
			Instance instance = m_Instances.instance(i);
			dParameters_.update(instance);
		}

		// Convert counts to Probability
		xxyDist_.countsToProbs();
		dParameters_.countsToProbability(xxyDist_);

		// System.out.println();
		// for (int c = 0; c < nc; c++) {
		// System.out.print(xxyDist_.xyDist_.getClassCount(c) + ", ");
		// }
		// System.out.println();
		// for (int u = 0; u < nAttributes; u++) {
		// for (int uval = 0; uval < paramsPerAtt[u]; uval++) {
		// for (int c = 0; c < nc; c++) {
		// System.out.print("P(x_" + u + "=" + uval + " | y = " + c + ") = "
		// + xxyDist_.xyDist_.getCount(u, uval, c) + ", ");
		// }
		// System.out.println();
		// }
		// }

		// free up some space
		m_Instances = null;
	}

	@Override
	public double[] distributionForInstance(Instance instance) {

		double[] probs = new double[nc];
		double[] xyProbs = new double[nc];
		for (int c = 0; c < nc; c++) {
			probs[c] = xxyDist_.xyDist_.pp(c);
			xyProbs[c] = xxyDist_.xyDist_.pp(c);
		}

		
		double[] tempProbs = new double[nc];
		double w1 = 0.5;
		double pp = 0.0;
		double[] attProbs = new double[nAttributes];
		
		for (int u = 0; u < nAttributes; u++) {
			
			attProbs[u] = xxyDist_.xyDist_.p(u,(int) instance.value(u));//P(u)
			System.out.println("*********************************");
			System.out.println("p("+u+"): "+ attProbs[u]);
			System.out.println();
			
			wdBayesNode bNode = dParameters_.getBayesNode(instance, u);// leaf node
			
			for (int c = 0; c < nc; c++) {
				
				if (m_Parents[u] != null) {
					
					pp = xxyDist_.xyDist_.jointP(u, (int) instance.value(m_Parents[u][0]), c);//P(u,c)
					w1 = computeW1(pp); 
					System.out.println("y="+c+" : "+w1);
				
//					for (int i = 0; i < m_Parents[u].length; i++) {
//						
//						pp = SUtils.Laplace(
//								xxyDist_.xyDist_.getCount(m_Parents[u][i], (int) instance.value(m_Parents[u][i]), c),
//								(double) nInstances, nc);
//					}
				}else{
					w1 = computeW1(xyProbs[c]);//w1 = N*P(y)/(N*P(y)+N0)
					System.out.println("w1 : "+w1);
				}
				System.out.println(bNode.getXYCount((int) instance.value(u), c));
				tempProbs[c] = smooth(w1,bNode.getXYCount((int) instance.value(u), c),attProbs[u]);//smooth = w1*P(u|y)+w2*P(u)
				System.out.println(tempProbs[c]);
				probs[c] *= tempProbs[c];
			}
		}

		// for (int u = 0; u < nAttributes; u++) {
		// 		wdBayesNode bNode = dParameters_.getBayesNode(instance, u);//leaf
		// 		for (int c = 0; c < nc; c++) {
		// 			probs[c] *= bNode.getXYCount((int) instance.value(m_Order[u]), c);
		// 		}
		// }

		Utils.normalize(probs);
		System.out.println();
		for (int i = 0; i < probs.length; i++) {
			System.out.println(probs[i]);
		}

		return probs;
	}

	public double computeW1(double a) {
		return (nInstances * a) / (nInstances * a + N0);
	}

	public double smooth(double w1, double p1, double p2) {
		double w2 = 1 - w1;
		return w1 * p1 + w2 * p2;
	}

	public void setN0(int a) {
		N0 = a;
	}

	public double[] logDistributionForInstance_MAP(Instance instance) {

		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = xxyDist_.xyDist_.pp(c);
		}

		for (int u = 0; u < nAttributes; u++) {
			wdBayesNode bNode = dParameters_.getBayesNode(instance, u);
			for (int c = 0; c < nc; c++) {
				probs[c] += bNode.getXYCount((int) instance.value(m_Order[u]), c);
			}
		}

		SUtils.normalizeInLogDomain(probs);
		return probs;
	}

	public void setLaplacePseudocount(double m){
		laplacePseudocount = m;
	}
	public double getLaplacePseudocount(int m){
		return laplacePseudocount;
	}
	
	public static double laplace(double freq1, double freq2, double numValues) {
		double mEsti = (freq1 + laplacePseudocount) / (freq2 + laplacePseudocount*numValues);
		return mEsti;
	}
	
	public static void main(String[] argv) throws Exception {

		FileReader frData = new FileReader("weather.nominal.arff");
		Instances data = new Instances(frData);//
		data.setClassIndex(data.numAttributes() - 1);

		kDBPenny classifier = new kDBPenny();
		classifier.setLaplacePseudocount(0.5);

		classifier.setN0(5);
		
		classifier.buildClassifier(data);
		classifier.distributionForInstance(data.instance(0));

//		Evaluation eval = new Evaluation(data);
////		eval.crossValidateModel(classifier, data, 10, new Random(1));
//
//		System.out.println(eval.errorRate());
//		System.out.println(eval.rootMeanSquaredError());

		// for (int i = 0; i < 11; i++) {
		// double m = 0.1 * i;
		// SUtils.setLaplaceM(m);
		//
		// Evaluation eval = new Evaluation(data);
		// eval.crossValidateModel(classifier, data, 10, new Random(1));
		// // System.out.print("M of Laplace equals:"+SUtils.getLaplaceM());
		// System.out.println(eval.rootMeanSquaredError());
		// }

		// runClassifier(new kDBPenny(), argv);
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
		m_MVerb = Utils.getFlag('V', options);
		m_Discretization = Utils.getFlag('D', options);

		String MK = Utils.getOption('K', options);
		if (MK.length() != 0) {
			m_KDB = Integer.parseInt(MK);
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

	public int getNInstances() {
		return nInstances;
	}

	public int getNc() {
		return nc;
	}

	public xxyDist getXxyDist_() {
		return xxyDist_;
	}

	public wdBayesParametersTree getdParameters_() {
		return dParameters_;
	}

	public int[] getM_Order() {
		return m_Order;
	}

	public boolean isM_MVerb() {
		return m_MVerb;
	}

	public int getnAttributes() {
		return nAttributes;
	}
}
