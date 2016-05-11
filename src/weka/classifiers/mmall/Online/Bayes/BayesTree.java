package weka.classifiers.mmall.Online.Bayes;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;

import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class BayesTree {

	protected double[] classParameters;
	private long[] classCounts;
	private double[] classProbabilities;
	protected double[] classGradients;

	private int np;

	private BayesNode[] roots;
	private int n;
	private int nClassValues;
	private long nInstances;

	protected int[] nParamsPerAtt;

	private int[] order;
	private int[][] parents;

	private ParamScheme scheme;

	private double initParameters;
	private double centerWeights;

	public double getCenterWeights() {
		return centerWeights;
	}

	public void setCenterWeights(double centerWeights) {
		this.centerWeights = centerWeights;
	}

	/**
	 * Constructor called by wdBayes
	 */
	public BayesTree(int n, int nc, int[] paramsPerAtt, int[] m_Order, int[][] m_Parents, ParamScheme scheme) {
		this.n = n;
		this.nClassValues = nc;
		this.scheme = scheme;

		nParamsPerAtt = new int[n];
		for (int u = 0; u < n; u++) {
			nParamsPerAtt[u] = paramsPerAtt[u];
		}

		order = new int[n];
		parents = new int[n][];

		for (int u = 0; u < n; u++) {
			order[u] = m_Order[u];
		}

		for (int u = 0; u < n; u++) {
			if (m_Parents[u] != null) {
				parents[u] = new int[m_Parents[u].length];
				for (int p = 0; p < m_Parents[u].length; p++) {
					parents[u][p] = m_Parents[u][p];
				}
			}
		}

		roots = new BayesNode[n];
		for (int u = 0; u < n; u++) {
			roots[u] = new BayesNode(m_Order[u], this);
		}

		classCounts = new long[nClassValues];
		classProbabilities = new double[nClassValues];
		classParameters = new double[nClassValues];
		classGradients = new double[nClassValues];
	}

	public void update(Instance instance) {
		updateCountClass(instance);
		for (int u = 0; u < n; u++) {
			roots[u].updateSubTreeWithNewInstance(instance, parents[u]);
		}
	}

	public void unupdate(Instance instance) {
		unupdateCountClass(instance);
		for (int u = 0; u < n; u++) {
			roots[u].unupdateSubTreeWithNewInstance(instance, parents[u]);
		}
	}

	private void updateCountClass(Instance instance) {
		int xC = (int) instance.classValue();
		classCounts[xC]++;
	}

	private void unupdateCountClass(Instance instance) {
		int xC = (int) instance.classValue();
		classCounts[xC]--;
	}

	protected void computeClassProbabilitiesFromCounts() {
		this.classProbabilities = new double[nClassValues];
		long totalCount = 0L;
		for (int c = 0; c < nClassValues; c++) {
			totalCount += classCounts[c];
		}
		for (int c = 0; c < nClassValues; c++) {
			double prob = Math.log(Math.max(SUtils.MEsti(classCounts[c], totalCount, nClassValues), 1e-75));
			classProbabilities[c] = prob;
		}
	}

	public void computeProbabilitiesFromCounts() {
		computeClassProbabilitiesFromCounts();
		for (int u = 0; u < n; u++) {
			roots[u].computeProbabilitiesFromCounts();
		}
	}

	public void setInitParametersToMAPEstimates() {
		setClassParametersToMAPEstimates();
		for (int u = 0; u < n; u++) {
			roots[u].setInitParametersToMAPEstimates();
		}

	}

	private void setClassParametersToMAPEstimates() {
		for (int c = 0; c < nClassValues; c++) {
			classParameters[c] = classProbabilities[c];
		}
	}

	public void allocateMemoryForParametersAndGradients() {
		for (int u = 0; u < n; u++) {
			roots[u].allocateMemoryForParametersAndGradients();
		}
	}

	public double getClassParameter(int c) {
		return classParameters[c];
	}

	public long getClassCount(int c) {
		return classCounts[c];
	}

	public double getClassProbability(int c) {
		return classProbabilities[c];
	}

	public int getNp() {
		return np;
	}

	public int getNAttributes() {
		return n;
	}

	public void computeGradientForClass_w(Instance instance, double[] probs, RegularizationType regType, double lambda, double center) {
		int x_C = (int) instance.classValue();
		for (int c = 0; c < nClassValues; c++) {
			classGradients[c] = (-1) * (SUtils.ind(c, x_C) - probs[c]) * classProbabilities[c];
			switch (regType) {
			case L1:
				classGradients[c] += lambda / 2.0;
				break;
			case L2:
				classGradients[c] += lambda * (classParameters[c] - center);
				break;
			case None:
				break;
			default:
				System.err.println("Regularization " + regType.name() + " not programmed");
			}
		}
	}

	public void computeGradientForClass_d(Instance instance, double[] probs, RegularizationType regType, double lambda, double center) {
		int x_C = (int) instance.classValue();
		for (int c = 0; c < nClassValues; c++) {
			classGradients[c] = (-1) * (SUtils.ind(c, x_C) - probs[c]);
			switch (regType) {
			case L1:
				classGradients[c] += lambda / 2.0;
				break;
			case L2:
				classGradients[c] += lambda * (classParameters[c] - center);
				break;
			case None:
				break;
			default:
				System.err.println("Regularization " + regType.name() + " not programmed");
			}
		}
	}

	public void updateParametersForClass(Instance instance, int t, double[] eta0, RegularizationType regularizationType, double lambda) {
		for (int c = 0; c < nClassValues; c++) {
			double step = 0.0;
			switch (regularizationType) {
			case L1:
			case L2:
				step = eta0[c] / (1 + eta0[c] * lambda * t);
				break;
			case None:
			default:
				step = eta0[c] / (1 + t);
			}
			classParameters[c] = classParameters[c] - step * classGradients[c];
		}

	}

	public int getNValuesForAttribute(int attNum) {
		return nParamsPerAtt[attNum];
	}

	public int getNClassValues() {
		return nClassValues;
	}

	public void resetGradientsForClass() {
		Arrays.fill(classGradients, 0.0);
	}

	public BayesNode[] findLeavesForInstance(final Instance instance) {
		BayesNode[] nodes = new BayesNode[getNAttributes()];
		for (int u = 0; u < nodes.length; u++) {
			nodes[u] = roots[u].getLeafForInstance(instance);
		}
		return nodes;
	}

	public void computeFinalProbabilities() {
		computeNInstances();
		for (int u = 0; u < roots.length; u++) {
			roots[u].computeFinalProbabilities();
		}
	}

	public void resetParameters() {
		Arrays.fill(classParameters, initParameters);
		for (int u = 0; u < roots.length; u++) {
			roots[u].resetParameters();
		}
	}

	public ParamScheme getScheme() {
		return scheme;
	}

	public String toString() {
		String str = "root\n";
		for (int u = 0; u < roots.length; u++) {
			str += roots[u].toString("\t");
		}
		return str;
	}

	public void setInitParameters(double initParameters) {
		this.initParameters = initParameters;
	}

	public double getInitParameters() {
		return initParameters;
	}

	public double l2NormShiftedLeaves() {
		double res = 0.0;
		for (int i = 0; i < classParameters.length; i++) {
			res += (classParameters[i] - centerWeights) * (classParameters[i] - centerWeights);
		}
		for (int u = 0; u < roots.length; u++) {
			res += roots[u].squaredl2normShiftedLeaves();
		}
		Math.sqrt(res);
		return res;
	}

	public double l1NormShiftedLeaves() {
		double res = 0.0;
		for (int i = 0; i < classParameters.length; i++) {
			res += Math.abs(classParameters[i] - centerWeights);
		}
		for (int u = 0; u < roots.length; u++) {
			res += roots[u].squaredl2normShiftedLeaves();
		}
		return res;
	}

	public long getNInstances() {
		return nInstances;
	}

	private void computeNInstances() {
		nInstances = 0L;
		for (int i = 0; i < classCounts.length; i++) {
			nInstances += classCounts[i];
		}
	}

	public void optimizeSmoothingParameter(Instances sample, wdBayesOnline algo) throws FileNotFoundException, IOException {
		// remove from model
		for (Instance instance : sample) {
			unupdate(instance);
		}
		computeProbabilitiesFromCounts();

		double bestSmoothingParam = 0.0;
		double bestRMSE = Double.MAX_VALUE;

		for (BayesNode.CONFIDENCE_PRIOR = 0; BayesNode.CONFIDENCE_PRIOR < 10; BayesNode.CONFIDENCE_PRIOR++) {

			// compute smoothed probabilities
			computeFinalProbabilities();
			int nDataPoints = 0;
			double RMSE = 0.0;
			for (Instance instance : sample) {
				int classValue = (int) instance.classValue();
				double[] probs = algo.distributionForInstance(instance);
				boolean someNaN = false;
				for (int c = 0; !someNaN && c < probs.length; c++) {
					if(Double.isNaN(probs[c])){
						someNaN = true;
					}
				}
				if(!someNaN){
					for (int i = 0; i < probs.length; i++) {
						if (i == classValue) {
							RMSE += (1.0 - probs[i]) * (1.0 - probs[i]);
						} else {
							RMSE += (0.0 - probs[i]) * (0.0 - probs[i]);
						}
					}
					nDataPoints++;
				}
			}
			RMSE = Math.sqrt(RMSE / nDataPoints);
			System.out.println("smooth param="+BayesNode.CONFIDENCE_PRIOR+" RMSE="+RMSE+" (best="+bestRMSE+")");
			if (RMSE < bestRMSE) {
				bestRMSE = RMSE;
				bestSmoothingParam = BayesNode.CONFIDENCE_PRIOR;
			}

		}
		BayesNode.CONFIDENCE_PRIOR = bestSmoothingParam;
		// put back into model
		for (Instance instance : sample) {
			update(instance);
		}
		computeProbabilitiesFromCounts();
		System.out.println("Selecting smoothing param="+BayesNode.CONFIDENCE_PRIOR);

	}

}