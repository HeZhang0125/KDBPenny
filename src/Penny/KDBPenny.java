package Penny;

import java.io.File;
import java.io.FileReader;
import java.util.Arrays;
import java.util.Random;
import org.apache.commons.math3.util.FastMath;
import weka.classifiers.*;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.mmall.DataStructure.xxyDist;
import weka.classifiers.mmall.Utils.CorrelationMeasures;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.filters.supervised.attribute.Discretize;

public class KDBPenny extends AbstractClassifier implements OptionHandler {
	private static final long serialVersionUID = 691858787988950382L;

	public static int nInstances;
	public static Instances m_Instances;
	int nAttributes;
	int nc;
	public int[] paramsPerAtt; // num of att values of each attribute
	xxyDist xxyDist_;

	private Discretize m_Disc = null;

	public wdBayesParametersTreeHPYP dParameters_;

	private int[][] m_Parents;
	private int[] m_Order;

	private boolean m_MVerb = false; // -V
	private boolean m_Discretization = false; // -D

	public int m_KDB = 1; // -K
	double[] proY;
	

	@Override
	public void buildClassifier(Instances Instances) throws Exception {

		m_Instances = Instances;

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
		proY = new double[nc];

		// paramsPerAtt[i]: the num of ith attribute values
		paramsPerAtt = new int[nAttributes + 1];
		for (int u = 0; u < paramsPerAtt.length; u++) {
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
		 * Initialize KDB structure
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

		// add y as its first parent
		for (int u = 0; u < nAttributes; u++) {
			if (m_Parents[u] == null) {
				m_Parents[u] = new int[1];
				m_Parents[u][0] = m_Instances.classIndex();
			} else {
				int[] temp = new int[m_Parents[u].length + 1];
				temp[0] = m_Instances.classIndex();

				for (int j = 0; j < m_Parents[u].length; j++)
					temp[j + 1] = m_Parents[u][j];
				m_Parents[u] = temp;
			}
		}

		// Print the structure
		printStructure();

		dParameters_ = new wdBayesParametersTreeHPYP(m_Instances, paramsPerAtt, m_Order, m_Parents);
		
		int[] yNum = new int[this.nc];
		
		// Update dTree_ based on parents
		for (int i = 0; i < nInstances; i++) {
			Instance instance = m_Instances.instance(i);
			yNum[(int)instance.classValue()]++;
			
			dParameters_.update(instance);
		}
		
		for(int i = 0; i < nc; i++){
			proY[i] = (double)yNum[i]/nInstances;
		}
//		System.out.println(Arrays.toString(yNum));
		
//		double discount = 0.0;
//		LogStirlingCache lgCache = new LogStirlingCache(discount, nInstances);
		
//		dParameters_.setLgCache(lgCache);
//		dParameters_.setDiscount(discount);
		
		dParameters_.probabilitySmooth();

		// free up some space
		m_Instances = null;

	}

	@Override
	public double[] distributionForInstance(Instance instance) {
		double[] probs = new double[nc];
		
		probs = this.proY;

		for (int y = 0; y < nc; y++) {
			System.out.print(probs[y]+"\t");
			
			for (int u = 0; u < nAttributes; u++) {
				wdBayesNode bNode = dParameters_.getBayesNode(instance, u, y);// leaf
//				probs[y] += FastMath.log(bNode.pkAccumulated[(int)instance.value(m_Order[u])]);
				 System.out.print(Arrays.toString(bNode.pkAccumulated));
//				 System.out.print(bNode.pkAccumulated[(int)instance.value(m_Order[u])]+"\t");
				 probs[y] *= bNode.pkAccumulated[(int)instance.value(m_Order[u])];
			}
			System.out.println();
			 System.out.println("probs["+y+"]: "+probs[y]);
		}

		
//		SUtils.normalizeInLogDomain(probs);
//		SUtils.exp(probs);
		return probs;
	}

	public static void main(String[] argv) throws Exception {

		File folder = new File("data");
		File[] listOfFiles = folder.listFiles();

		for (int n = 0; n < 1; n++) {
			if (listOfFiles[n].isFile()) {

				FileReader frData = new FileReader(listOfFiles[n]);
				Instances data = new Instances(frData);
				data.setClassIndex(data.numAttributes() - 1);
				System.out.println(listOfFiles[n].getName());
				System.out.println(data.numAttributes());

				KDBPenny classifier = new KDBPenny();
				classifier.setKDB(1);

				classifier.buildClassifier(data);
//				classifier.distributionForInstance(data.firstInstance());
//				int iteration =5;
//				double brierScore = 0, error = 0;
//				for (int i = 0; i < iteration; i++) {
//					// System.out.println("****************" + i);
//					Evaluation eva = new Evaluation(data);
//					eva.crossValidateModel(classifier, data, 2, new Random());
//					brierScore += (eva.rootMeanSquaredError() * eva.rootMeanSquaredError());
//					error += eva.errorRate();
//					// System.out.println("****************" + i + "
//					// done");
//				}
//				brierScore /= iteration;
//				error /= iteration;
//				System.out.println(brierScore + "\t" + error);
			}
		}
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

	public wdBayesParametersTreeHPYP getdParameters_() {
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

	public void setKDB(int k) {
		m_KDB = k;
	}

	public int getKDB() {
		return m_KDB;
	}

	public void printStructure() {
		for (int i = 0; i < nAttributes; i++) {
			System.out.print(i + " : ");
			if (m_Parents[i] != null) {
				for (int j = 0; j < m_Parents[i].length; j++) {
					System.out.print(m_Parents[i][j] + ",");
				}
			}
			System.out.println();
		}
	}

	public void printParameters() {

		System.out.println(this.m_KDB);
		System.out.println(this.dParameters_.toString());
	}
}
