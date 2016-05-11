package weka.classifiers.mmall.Bayes.objectiveFunction;

import java.util.Arrays;

import lbfgsb.FunctionValues;
import weka.classifiers.mmall.Bayes.wdBayes;
import weka.classifiers.mmall.DataStructure.Bayes.wdBayesNode;
import weka.classifiers.mmall.DataStructure.Bayes.wdBayesParametersTree;
import weka.classifiers.mmall.DataStructure.xyDist;
import weka.classifiers.mmall.Utils.SUtils;
import weka.core.Instance;
import weka.core.Instances;

public class ObjectiveFunctionCLL_w extends ObjectiveFunctionCLL {

	public ObjectiveFunctionCLL_w(wdBayes algorithm) {
		super(algorithm);
	}

	@Override
	public FunctionValues getValues(double params[]) {

		//System.out.println("in getValues()");
		//System.out.println(Arrays.toString(params));
		
		double negLogLikelihood = 0.0;
		algorithm.dParameters_.copyParameters(params);
		double g[] = new double[algorithm.dParameters_.getNp()];

		int N = algorithm.getNInstances();
		int n = algorithm.getnAttributes();
		int nc = algorithm.getNc();
		double[] myProbs = new double[nc];
		
		wdBayesParametersTree dParameters = algorithm.getdParameters_();
		Instances instances = algorithm.getM_Instances();
		wdBayesNode[] myNodes = new wdBayesNode[n];
		
		int[] order = algorithm.getM_Order();
		double mLogNC = -Math.log(nc);
		
		xyDist xyDist = algorithm.getXxyDist_().xyDist_;
		
		boolean m_Regularization = algorithm.getRegularization();
		double m_Lambda = algorithm.getLambda();

		for (int i = 0; i < N; i++) {
			Instance instance = instances.instance(i);

			int x_C = (int) instance.classValue();

			wdBayes.findNodesForInstance(myNodes, instance, dParameters);

			// unboxed logDistributionForInstance_w
			for (int c = 0; c < nc; c++) {
				myProbs[c] = xyDist.pp(c) * dParameters.getClassParameter(c);
			}

			for (int c = 0; c < nc; c++) {
				for (int u = 0; u < myNodes.length; u++) {
					wdBayesNode bNode = myNodes[u];
					myProbs[c] += bNode.getXYParameter((int) instance.value(order[u]), c) * bNode.getXYCount((int) instance.value(order[u]), c);
				}
			}

			SUtils.normalizeInLogDomain(myProbs);
			negLogLikelihood += (mLogNC - myProbs[x_C]);
			SUtils.exp(myProbs);

			// unboxed logGradientForInstance_w
			for (int c = 0; c < nc; c++) {
				if (m_Regularization) {
					negLogLikelihood += m_Lambda/2 * dParameters.getClassParameter(c) * dParameters.getClassParameter(c);
					g[c] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * xyDist.pp(c)  + m_Lambda * dParameters.getClassParameter(c);
				} else {
					g[c] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * xyDist.pp(c);
				}
			}

			for (int u = 0; u < myNodes.length; u++) {
				wdBayesNode bayesNode = myNodes[u];
				for (int c = 0; c < nc; c++) {
					int index = bayesNode.getXYIndex((int) instance.value(order[u]), c);
					double probability = bayesNode.getXYCount((int) instance.value(order[u]), c);
					double parameter = bayesNode.getXYParameter((int) instance.value(order[u]), c);
					
					if (m_Regularization) {
						negLogLikelihood += m_Lambda/2 * parameter * parameter;
						g[index] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * probability + m_Lambda * parameter;
					} else {
						g[index] += (-1) * (SUtils.ind(c, x_C) - myProbs[c]) * probability;
					}
				}
			}
		}

//		if (algorithm.isM_MVerb()) {
//			System.out.print(negLogLikelihood + ", ");
//		}

		FunctionValues fv = new FunctionValues(negLogLikelihood, g);
		//System.out.println("fv="+fv.functionValue+"\t"+negLogLikelihood);
		return fv;
	}

}