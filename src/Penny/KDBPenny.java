package Penny;

import java.io.File;
import java.io.FileReader;
import java.io.PrintStream;
import java.util.Random;
import weka.classifiers.*;
import org.apache.commons.math3.util.FastMath;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.mmall.DataStructure.xxyDist;
import weka.classifiers.mmall.DataStructure.Bayes.wdBayesNode;
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

	public wdBayesParametersTree dParameters_;

	private int[][] m_Parents;
	private int[] m_Order;

	private boolean m_MVerb = false; // -V
	private boolean m_Discretization = false; // -D

	public int m_KDB = 1; // -K
	public static double N0 = 5;
	public static double laplacePseudocount = 0.5;

	public enum Method {
		LAPLACE, TAN, Dirichlet
	};

	public static Method method = Method.LAPLACE;

	public enum MethodN0 {
		N05, AutoN0, FunctionN0
	};

	public static MethodN0 N0Indicator = MethodN0.N05;

	public enum MethodM {
		ConstantM, AutoM
	};

	public static MethodM MIndicator = MethodM.ConstantM;

	public static double[] m_alpha = { 0.5, 0.5, 0.5, 0.5, 0.5 };

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

		// Print the structure
		// printStructure();

		switch (method) {
		case LAPLACE: {
			dParameters_ = new wdBayesParametersTreeLaplace(m_Instances, paramsPerAtt, m_Order, m_Parents, 1);

			// if (MIndicator == MethodM.AutoM) {
			// double[] valueM = { 0, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 2, 3, 4, 5,
			// 10, 20 };
			// double bestM = autoFindBestParameter(this, valueM, true);
			//// System.out.println("bestM: "+ bestM);
			// setLaplacePseudocount(bestM); // set M to best M
			// }
			break;
		}
		case TAN: {
			dParameters_ = new wdBayesParametersTreeTAN(m_Instances, paramsPerAtt, m_Order, m_Parents, 1);

			double[] valueN0 = { 0, 0.5, 1, 1.5, 2, 5, 10, 15, 20, 50, 100, 200, 1000 };

			if (N0Indicator == MethodN0.N05) {
				dParameters_.setFunctionFlag(false);
				dParameters_.setN0(5);
				setN0(5);
			}
			if (N0Indicator == MethodN0.AutoN0) {
				dParameters_.setFunctionFlag(false);
				double bestN0 = autoFindBestParameter(this, valueN0);
				dParameters_.setN0(bestN0); // set N0 to best N0
				setN0(bestN0);
			}

			if (N0Indicator == MethodN0.FunctionN0) {
				dParameters_.setFunctionFlag(true);
				computeAlpha(this, valueN0);
				dParameters_.setAlpha(m_alpha);
			}
			break;
		}
		case Dirichlet: {
			dParameters_ = new wdBayesParametersTreeDirichlet(m_Instances, paramsPerAtt, m_Order, m_Parents, 1);

			double[] valueN0 = { 0, 0.5, 1, 1.5, 2, 5, 10, 15, 20, 50, 100, 200, 1000 };
			if (N0Indicator == MethodN0.N05) {
				dParameters_.setFunctionFlag(false);
				dParameters_.setN0(5);
				setN0(5);
			}
			if (N0Indicator == MethodN0.AutoN0) {
				dParameters_.setFunctionFlag(false);
				double bestN0 = autoFindBestParameter(this, valueN0);
				dParameters_.setN0(bestN0); // set N0 to best N0
				setN0(bestN0);
			}
			if (N0Indicator == MethodN0.FunctionN0) {
				dParameters_.setFunctionFlag(true);
				computeAlpha(this, valueN0);
				dParameters_.setAlpha(m_alpha);
			}
			break;
		}
		}

		// Update dTree_ based on parents
		for (int i = 0; i < nInstances; i++) {
			Instance instance = m_Instances.instance(i);
			dParameters_.update(instance);
		}

		// Convert counts to Probability
		xxyDist_.countsToProbs();
		dParameters_.countsToProbability(xxyDist_);

		// free up some space
		m_Instances = null;
	}

	@Override
	public double[] distributionForInstance(Instance instance) {
		double[] probs = new double[nc];

		for (int c = 0; c < nc; c++) {
			probs[c] = FastMath.log(xxyDist_.xyDist_.pp(c));// P(y)
		}

		for (int u = 0; u < nAttributes; u++) {
			wdBayesNode bNode = dParameters_.getBayesNode(instance, u);// leaf
			for (int y = 0; y < nc; y++) {
				probs[y] += FastMath.log(bNode.getXYProbability((int) instance.value(m_Order[u]), y));
			}
		}

		SUtils.normalizeInLogDomain(probs);
		SUtils.exp(probs);
		return probs;
	}

	public static void main(String[] argv) throws Exception {

		File folder = new File("C:/Users/zhe/workspace/Penny/datasets");
		File[] listOfFiles = folder.listFiles();

		for (int n = 0; n < 1; n++) {
			if (listOfFiles[n].isFile()) {

				FileReader frData = new FileReader(listOfFiles[n]);
				Instances data = new Instances(frData);
				data.setClassIndex(data.numAttributes() - 1);
				System.out.println(listOfFiles[n].getName());
				System.out.println(data.numAttributes());

				 PrintStream myconsole = new PrintStream(new File(listOfFiles[n].getName() + ".txt"));
				 System.setOut(myconsole);

				for (int k = 0; k < 6; k++) {
					System.out.println("******************************K=" + k);
					// run experiment for 5 times

					KDBPenny classifier = new KDBPenny();
					classifier.setKDB(k);
					// experiments: Laplace, TAN, Dirichlet
					for (Method m : Method.values()) {

						classifier.setMethodFlag(m);
//						System.out.println("^^^^^^^^^^^^^^^" + classifier.getMethodFlag().toString());

						if (method == Method.LAPLACE) {

//							double[] value = { 0, 0.5, 1, 1.5, 2, 5, 10, 15, 20 };
//
//							double minRMSE = Double.MAX_VALUE;
//							for (int p = 0; p < value.length; p++) {
//								Evaluation eval = new Evaluation(data);
//								classifier.setLaplacePseudocount(value[p]);
//
//								eval.crossValidateModel(classifier, data, 5, new Random(1));
//								double rmse = eval.rootMeanSquaredError();
//								if (p < 3) {
//									System.out.println(rmse);
//								}
//
//								if (rmse < minRMSE) {
//									minRMSE = rmse;
//								}
//							}
//							System.out.println(minRMSE);

						} else if (method == Method.TAN || method == Method.Dirichlet) {

							classifier.setLaplacePseudocount(0.5);
							for (MethodN0 w : MethodN0.values()) {

								classifier.setN0Flag(w);

								if (w == MethodN0.N05 || w == MethodN0.AutoN0) {

									Evaluation eval = new Evaluation(data);
									eval.crossValidateModel(classifier, data, 5, new Random(1));
									double rmse = eval.rootMeanSquaredError();
									System.out.println(rmse);
								} else if (w == MethodN0.FunctionN0) {
									double sumRMSE = 0;
									for (int a = 0; a < 5; a++) {
										Evaluation eval = new Evaluation(data);
										eval.crossValidateModel(classifier, data, 5, new Random(1));
										double rmse = eval.rootMeanSquaredError();
										sumRMSE += rmse;
									}
									System.out.println(sumRMSE / 5);
								}
							}
						}
					}
				}
			}
		}
	}

	public static void computeAlpha(KDBPenny classifier, double[] value) throws Exception {

		for (int j = 0; j < 10; j++) {

			Instances train = m_Instances.trainCV(10, j, new Random());// train.subtrain
			Instances test = m_Instances.testCV(10, j);// train.subtest
			Evaluation eval = new Evaluation(test);
			double minRMSE = Double.MAX_VALUE;

			// update counts
			for (int k = 0; k < train.numInstances(); k++) {
				Instance instance = train.instance(k);
				classifier.dParameters_.update(instance);
			}

			SlidingWindow window = new SlidingWindow(5);

			for (int i = 0; i < m_alpha.length; i++) {
				for (int z = 0; z < value.length; z++) {

					// find the slidingWindow of alpha[i]
					if (value[z] == m_alpha[i]) {

						if (z == 0) {
							window.put(value[z]);
							window.put(value[z + 1]);
							window.put(value[z + 2]);
						} else if (z == 1) {
							window.put(value[z - 1]);
							window.put(value[z]);
							window.put(value[z + 1]);
							window.put(value[z + 2]);
						} else if (z == 11) {
							window.put(value[z - 2]);
							window.put(value[z - 1]);
							window.put(value[z]);
							window.put(value[z + 1]);
						} else if (z == 12) {
							window.put(value[z - 2]);
							window.put(value[z - 1]);
							window.put(value[z]);
						} else {
							window.put(value[z - 2]);
							window.put(value[z - 1]);
							window.put(value[z]);
							window.put(value[z + 1]);
							window.put(value[z + 2]);
						}

						double bestParameter = m_alpha[i];

						for (int p = 0; p < window.windowsize; p++) {
							m_alpha[i] = window.get(p);

							classifier.setAlpha(m_alpha);// set alpha
							classifier.dParameters_.setAlpha(m_alpha);

							// compute smoothed probability
							classifier.xxyDist_.countsToProbs();
							classifier.dParameters_.countsToProbability(classifier.xxyDist_);

							// evaluate RMSE on train.subtest
							eval.evaluateModel(classifier, test);
							double rmse = eval.rootMeanSquaredError();

							if (rmse < minRMSE) {
								minRMSE = rmse;
								bestParameter = window.get(p);
							}
						}
						m_alpha[i] = bestParameter;
						break;
					}
				}
			}
			classifier.dParameters_.reseCountsToZero();
		}
	}

	// auto find the best value of M or N0
	public static double autoFindBestParameter(KDBPenny classifier, double[] value) throws Exception {

		double bestParameter = 0;
		double minRMSE = Double.MAX_VALUE;
		// 10 folds cross validation
		for (int i = 0; i < 10; i++) {

			Instances train = m_Instances.trainCV(10, i, new Random());// train.subtrain
			Instances test = m_Instances.testCV(10, i);// train.subtest
			Evaluation eval = new Evaluation(test);

			// update counts
			for (int k = 0; k < train.numInstances(); k++) {
				Instance instance = train.instance(k);
				classifier.dParameters_.update(instance);
			}

			for (int j = 0; j < value.length; j++) {
				classifier.setN0(value[j]); // set N0
				classifier.dParameters_.setN0(value[j]);
				
				// compute smoothed probability
				classifier.xxyDist_.countsToProbs();
				classifier.dParameters_.countsToProbability(classifier.xxyDist_);

				// evaluate RMSE on train.subtest
				eval.evaluateModel(classifier, test);
				double rmse = eval.rootMeanSquaredError();

				if (rmse < minRMSE) {
					minRMSE = rmse;
					bestParameter = value[j];
				}
			}
			classifier.dParameters_.reseCountsToZero();
		}
		return bestParameter;// remember the best N0
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

	public void setKDB(int k) {
		m_KDB = k;
	}

	public int getKDB() {
		return m_KDB;
	}

	public void setLaplacePseudocount(double m) {
		laplacePseudocount = m;
	}

	public double getLaplacePseudocount() {
		return laplacePseudocount;
	}

	public static double laplace(double freq1, double freq2, double numValues) {
		double mEsti = (freq1 + laplacePseudocount) / (freq2 + laplacePseudocount * numValues);
		return mEsti;
	}

	public void setMethodFlag(Method flag) {
		method = flag;
	}

	public Method getMethodFlag() {
		return method;
	}

	public Method getFlag() {
		return method;
	}

	public void setAlpha(double[] alpha) {
		for (int i = 0; i < alpha.length; i++)
			m_alpha[i] = alpha[i];
	}

	public void setN0Flag(MethodN0 flag) {
		N0Indicator = flag;
	}

	public MethodN0 getN0Flag() {
		return N0Indicator;
	}

	public void setMFlag(MethodM flag) {
		MIndicator = flag;
	}

	public void setN0(double n0) {
		N0 = n0;
	}

	public double getN0() {
		return N0;
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
		System.out.println(this.getLaplacePseudocount());
		System.out.println(this.getN0());
		System.out.println(this.dParameters_.toString());
	}
}
